import os
import wandb
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from libs.loss import dice_loss
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from libs.alaskaNKDataloader import AlaskaNKMetaDataset
from libs.attentionUnet import ChannelAttention, SpatialAttention
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, jaccard_score


class MAMLTrainer:
    def __init__(self, base_model, episodes, config, args):
        self.base_model = base_model
        self.episodes = episodes
        self.config = config
        self.args = args
        self.inner_lr = config['inner_lr']
        self.meta_lr = config['meta_lr']
        self.meta_optimizer = tf.keras.optimizers.Adam(learning_rate=self.meta_lr)
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.inner_lr_no_improvement_count = 0
        self.meta_lr_no_improvement_count = 0
        self._prepare_exp()

    def get_run_name(self):
        return self.args.run_name
    
    def run_output_dir(self):
        return self.args.run_model_save_dir

    def _prepare_exp(self):
        # create run_name 
        date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")   
        self.args.run_name = f'maml_{self.args.num_samples_per_location}samples_{self.args.num_episodes}eps_{date_time}'

        # create save paths
        self.args.run_model_save_dir = os.path.join(self.args.save_path, self.args.run_name)
        os.makedirs(self.args.run_model_save_dir, exist_ok=True)

    # Function to normalize data based on the specified normalization type
    def _normalize_data(self, data, normalization_type='-1'):
        """Normalizes data based on the specified normalization type."""
        if normalization_type == '0':
            data_min = 0
            data_max = 255
            return (data - data_min) / (data_max - data_min)
        elif normalization_type == '-1':
            data_min = 0
            data_max = 255
            return 2 * ((data - data_min) / (data_max - data_min)) - 1
        elif normalization_type == 'none':
            return data
        else:
            raise ValueError("Unsupported normalization type. Choose '0', '-1', or 'none'.")


    def initialize_wandb(self, run_name):
        wandb.init(project=self.args.wandb_project_name, name=run_name, group=self.args.run_name, config=self.config)

    """////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    //////////////////// Train initial MAML model /////////////////////
    ///////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////"""
    def _train_task_model(self, model, inputs, outputs, optimizer):
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = dice_loss(outputs, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return model, loss.numpy()

    def train(self):
        print(f'Initialize the Training process: {self.args.run_name}')
        self.initialize_wandb(run_name = self.args.run_name)

        inner_lr_patience = self.config['patience'] - 5
        inner_lr_reduction_factor = 0.96
        meta_lr_patience = self.config['patience']
        meta_lr_reduction_factor = 0.96

        for epoch in range(self.config['epochs']):
            task_losses = []
            task_updates = []
            for episode_index, episode in enumerate(self.episodes):
                model_copy = tf.keras.models.clone_model(self.base_model)
                model_copy.set_weights(self.base_model.get_weights())
                inner_optimizer = tf.keras.optimizers.Adam(learning_rate=self.inner_lr)

                support_data, support_labels = episode["support_set_data"], episode["support_set_labels"]
                inner_loss = []
                for _ in range(self.config['inner_steps']):
                    model_copy, loss = self._train_task_model(model_copy, support_data, support_labels, inner_optimizer)
                    inner_loss.append(loss)

                query_data, query_labels = episode["query_set_data"], episode["query_set_labels"]
                with tf.GradientTape() as meta_tape:
                    meta_tape.watch(model_copy.trainable_variables)
                    meta_loss = dice_loss(query_labels, model_copy(query_data))
                gradients = meta_tape.gradient(meta_loss, model_copy.trainable_variables)
                task_losses.append(meta_loss.numpy())

                wandb.log({
                    "epoch": epoch,
                    "episode": episode_index,
                    "avg_inner_loss": tf.reduce_mean(inner_loss),
                    "outer_loss": meta_loss.numpy(),
                    "inner_lr": self.inner_lr,
                })

                mapped_gradients = [tf.identity(grad) for grad in gradients]
                task_updates.append((mapped_gradients, meta_loss))

            if task_updates:
                num_variables = len(self.base_model.trainable_variables)
                mean_gradients = []
                for i in range(num_variables):
                    grads = [update[0][i] for update in task_updates if update[0][i] is not None]
                    if grads:
                        mean_grad = tf.reduce_mean(tf.stack(grads), axis=0)
                        mean_gradients.append(mean_grad)
                    else:
                        print(f"No gradients for variable {i}")
                        mean_gradients.append(None)

                gradients_to_apply = [(grad, var) for grad, var in zip(mean_gradients, self.base_model.trainable_variables) if grad is not None]
                if gradients_to_apply:
                    self.meta_optimizer.apply_gradients(gradients_to_apply)
                else:
                    print("No gradients to apply")

            mean_loss = tf.reduce_mean(task_losses)
            wandb.log({
                "epoch": epoch,
                "mean_val_loss": mean_loss,
                "meta_lr": self.meta_optimizer.learning_rate.numpy()
            })

            if mean_loss < self.best_loss:
                self.best_loss = mean_loss
                self.no_improvement_count = 0
                self.inner_lr_no_improvement_count = 0
                self.meta_lr_no_improvement_count = 0
                model_save_path = os.path.join(self.args.run_model_save_dir,"maml_train.keras")
                self.base_model.save(model_save_path)
                print(f"Saved new best model with validation loss: {self.best_loss}")
            else:
                self.no_improvement_count += 1
                self.inner_lr_no_improvement_count += 1
                self.meta_lr_no_improvement_count += 1
                print(f"No improvement in epoch {epoch + 1}. No improvement count: {self.no_improvement_count}")

            if self.inner_lr_no_improvement_count >= inner_lr_patience:
                old_inner_lr = self.inner_lr
                self.inner_lr = old_inner_lr * inner_lr_reduction_factor
                print(f"Reducing inner learning rate from {old_inner_lr} to {self.inner_lr}")
                self.inner_lr_no_improvement_count = 0

            if self.meta_lr_no_improvement_count >= meta_lr_patience:
                old_meta_lr = self.meta_optimizer.learning_rate.numpy()
                new_meta_lr = old_meta_lr * meta_lr_reduction_factor
                self.meta_optimizer.learning_rate.assign(new_meta_lr)
                print(f"Reducing meta learning rate from {old_meta_lr} to {new_meta_lr}")
                self.meta_lr_no_improvement_count = 0

            if self.no_improvement_count >= self.config['patience']:
                print(f"Early stopping triggered after {epoch + 1} epochs with best validation loss: {self.best_loss}")
                break

            tf.keras.backend.clear_session()
            print(f"Epoch {epoch + 1} completed, Mean Validation Loss across all episodes: {mean_loss}")

        print(f"Completed training for maximum {self.config['epochs']} epochs.")
        wandb.finish()
        return self.base_model, model_save_path

    """////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    ////////////// Adapt initial MAML model to clusters ///////////////
    ///////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////"""
    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss, 'ChannelAttention':ChannelAttention, 'SpatialAttention':SpatialAttention})
        print(f"Loaded model from {model_path}")
        return model
    
    # Function to extract scenario information from the CSV filename
    def _extract_scenario_from_filename(csv_filename):
        """Extracts scenario information from the CSV filename."""
        # huc_code_kmean_5_cluster_4_test.csv
        basename = os.path.basename(csv_filename)
        parts = basename.replace('.csv', '').split('_')
        scenario = {}
        if "random" in parts:
            scenario_type = "random"
            idx_random = parts.index("random")
            num_clusters = parts[idx_random + 1]
            cluster_id = parts[idx_random + 3]
        elif "kmean" in parts: 
            scenario_type = "kmean"
            idx_random = parts.index("kmean")
            num_clusters = parts[idx_random + 1]
            cluster_id = parts[idx_random + 3]
        else:
            scenario_type = "regular"
            idx_code = parts.index("code")
            num_clusters = parts[idx_code + 1]
            cluster_id = parts[idx_code + 3]
        scenario = {
            "type": scenario_type,
            "num_clusters": num_clusters,
            "cluster_id": cluster_id
        }
        return scenario
    # Function to plot the channels of an input patch, prediction, and label
    def _plot_patch_channels(self, X_test_patch, y_probs_patch, y_pred_patch, y_test_patch, save_path, channels):
        num_channels = len(channels)
        nrows = 2  # Fixed to 2 rows
        ncols = (num_channels + 3 + nrows - 1) // nrows  # Calculate the number of columns dynamically

        # Create a figure for the plot
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 8)) 
        
        # Plot each channel of the input patch
        for i in range(num_channels):
            ax = axes.flatten()[i]
            ax.imshow(X_test_patch[:, :, i], cmap='gray')
            ax.set_title(f'Channel {i}')
            ax.axis('off')
        
        # Plot the prediction
        axes.flatten()[num_channels].imshow(y_probs_patch, cmap='gray')
        axes.flatten()[num_channels].set_title('Probs')
        axes.flatten()[num_channels].axis('off')
        
        # Plot the label
        axes.flatten()[num_channels + 1].imshow(y_pred_patch, cmap='gray')
        axes.flatten()[num_channels + 1].set_title('Prediction')
        axes.flatten()[num_channels + 1].axis('off')

            # Plot the prediction
        axes.flatten()[num_channels + 2].imshow(y_test_patch, cmap='gray')
        axes.flatten()[num_channels + 2].set_title('Label')
        axes.flatten()[num_channels + 2].axis('off')
        
        # Save the figure as a PNG file
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    
    def load_data_and_labels_from_csv(self, csv_path, data_dir, normalization_type, num_sample_per_location, channels):
        """Loads and concatenates the data and label .npy files listed in the CSV, selecting specific channels."""
        import pandas as pd  # Import pandas locally, as it's used in this function
        df = pd.read_csv(csv_path)
        all_data = []
        all_labels = []
        
        for huc_code in df['huc_code']:
            data_path = os.path.join(data_dir, f'{huc_code}_data.npy')
            label_path = os.path.join(data_dir, f'{huc_code}_label.npy')
            
            if not os.path.exists(data_path) or not os.path.exists(label_path):
                raise FileNotFoundError(f"Missing data or label file for HUC code: {huc_code}")
            
            # Load the data and labels
            data = np.load(data_path)[..., channels]  # Select only the specified channels
            data = self._normalize_data(data, normalization_type)  # Normalize data
            labels = np.load(label_path)

            # Sample data (or use all if samples_per_huc_code is -1)
            if num_sample_per_location == -1:
                sampled_data = data
                sampled_labels = labels
            else:
                indices = np.random.choice(data.shape[0], size=num_sample_per_location, replace=False)
                sampled_data = data[indices]
                sampled_labels = labels[indices]
            
            all_data.append(sampled_data)
            all_labels.append(sampled_labels)
            
            print(f"Loaded data and labels for HUC code: {huc_code}")

        # Concatenate all data and labels
        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)

        print(f"Combined dataset shape: {combined_data.shape}, Combined labels shape: {combined_labels.shape}")

        return combined_data, combined_labels

    def adapt_cluster_fit(self, csv_path, init_model_path, scenario, num_samples_per_location=-1):
        # Set the model save path for this cluster
        self.args.model_save_path = os.path.join(self.args.run_model_save_dir, f"cluster_{scenario['cluster_id']}_model")

        # Load the model
        model = self._load_model(init_model_path)

        # Compile the model with the optimizer and loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.adapt_learning_rate)
        model.compile(optimizer=optimizer, loss=dice_loss, metrics=[dice_loss])

        # Load the data and labels
        data, labels = self.load_data_and_labels_from_csv(
            csv_path, 
            self.args.data_dir, 
            self.args.normalization_type, 
            num_samples_per_location, 
            self.args.channels
        )

        # Split the data into training and validation sets
        train_data, val_data, train_labels, val_labels = train_test_split(
            data, labels, test_size=0.3, random_state=44
        )

        # Create callbacks for early stopping and model checkpoint
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=self.args.patience, 
            restore_best_weights=True
        )

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.args.model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            options=None
        )

        # Train the model using Keras's fit function
        history = model.fit(
            train_data, 
            train_labels, 
            validation_data=(val_data, val_labels), 
            batch_size=32,
            epochs=100,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        # Log training history to WandB
        for step, (train_loss, val_loss) in enumerate(zip(history.history["loss"], history.history["val_loss"])):
            wandb.log({
                "adapt_step": step + 1,
                "adapt_train_loss": train_loss,
                "adapt_val_loss": val_loss
            })

        print(f"Training completed. Best model saved at {self.args.model_save_path}")

        return model, self.args.model_save_path

    def adapt_cluster(self, csv_path, init_model_path, scenario, num_samples_per_location=-1):
        # Set the model save path for this cluster
        self.args.model_save_path = os.path.join(self.args.run_model_save_dir,f"cluster_{scenario['cluster_id']}_model.keras")

        # Load the model
        model = self._load_model(init_model_path)

        # Load the data and labels
        data, labels = self.load_data_and_labels_from_csv(csv_path, self.args.data_dir, self.args.normalization_type, num_samples_per_location, self.args.channels)
        train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.3, random_state=44)

        best_val_loss = float('inf')
        best_model = None
        no_improvement_count = 0  # Tracks how many steps without improvement
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.adapt_learning_rate)

        # Adaptation loop
        for step in range(self.args.adapt_steps):
            # Train on the training set
            with tf.GradientTape() as tape:
                predictions = model(train_data)
                adapt_loss = dice_loss(train_labels, predictions)
            gradients = tape.gradient(adapt_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Evaluate on the validation set
            val_predictions = model(val_data)
            val_loss = dice_loss(val_labels, val_predictions)

            # Log to wandb
            wandb.log({
                "adapt_step": step + 1,
                "adapt_train_loss": adapt_loss.numpy(),
                "adapt_val_loss": val_loss.numpy()
            })

            print(f"Step {step + 1}/{self.args.adapt_steps}: Training Loss = {adapt_loss.numpy()}, Validation Loss = {val_loss.numpy()}")

            # Check if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = tf.keras.models.clone_model(model)
                best_model.set_weights(model.get_weights())
                no_improvement_count = 0  # Reset the counter if validation improves
                print(f"New best model found at step {step + 1}, Validation Loss: {best_val_loss.numpy()}")
            else:
                no_improvement_count += 1  # Increment the no improvement counter

            # Early stopping condition
            if no_improvement_count >= self.args.patience:
                print(f"Early stopping at step {step + 1} due to no improvement in validation loss for {self.args.patience} steps.")
                break
                
        # Save the best model
        if best_model is not None:
            best_model.save(self.args.model_save_path)
            print(f"Best adapted model saved at {self.args.model_save_path}")

            # Log the model save path to WandB
            wandb.log({"adapt_model_save_path": self.args.model_save_path})

        else:
            print("No improvement in validation loss during training. No model saved.")

        return best_model, self.args.model_save_path
    
    def evaluate_clusters(self, model, test_csv_path, scenario, threshold=0.33): 
        # Set the metrics save path for this cluster
        self.args.metrics_save_path = os.path.join(self.args.run_model_save_dir,f"cluster_{scenario['cluster_id']}_eval.csv")

        # Load the CSV file containing huc_code list, only read the 'huc_code' column
        huc_codes = pd.read_csv(test_csv_path, usecols=['huc_code'])
        huc_codes_list = huc_codes['huc_code'].tolist()
        
        metrics = []
        table = wandb.Table(columns=["huc_code", "precision", "recall", "f1_score", "IoU", "kappa"])

        for huc_code in huc_codes_list:
            # Load the data and labels
            data_path = os.path.join(self.args.data_dir, f'{huc_code}_data.npy')
            label_path = os.path.join(self.args.data_dir, f'{huc_code}_label.npy')
            data = np.load(data_path)[..., self.args.channels]
            data = self._normalize_data(data, self.args.normalization_type)
            labels = np.load(label_path)

            y_pred_probs = model.predict(data)
            y_pred = (y_pred_probs > threshold).astype(int)

            # Select the 10th input patch, prediction, and label for plotting
            indx = 20
            if data.shape[0] >= indx:
                X_test_patch = data[indx]
                y_pred_probs_patch = y_pred_probs[indx]
                y_pred_patch = y_pred[indx]
                y_test_patch = labels[indx]
                
                # Save the plot of the 10th patch channels, prediction, and label
                plot_filename = os.path.join(self.args.run_model_save_dir, f'{huc_code}.png')
                self._plot_patch_channels(X_test_patch, y_pred_probs_patch, y_pred_patch, y_test_patch, plot_filename, self.args.channels)
            
            # Flatten the predictions and labels for pixel-level evaluation
            y_pred_flat = y_pred.flatten()
            y_test_flat = labels.flatten()
            
            # Calculate precision, recall, and F1-score for class 1
            f1_stream = f1_score(y_test_flat, y_pred_flat, labels=[1], average='micro')
            precision = precision_score(y_test_flat, y_pred_flat, pos_label=1, zero_division=0)
            recall = recall_score(y_test_flat, y_pred_flat, pos_label=1, zero_division=0)
            f1 = f1_score(y_test_flat, y_pred_flat, pos_label=1, zero_division=0)
            iou = jaccard_score(y_test_flat, y_pred_flat, pos_label=1, average='binary', zero_division=0)
            kappa = cohen_kappa_score(y_test_flat, y_pred_flat)

            # Add the metrics for this huc_code to the WandB table
            table.add_data(huc_code, precision, recall, f1, iou, kappa)
            print(f"HUC code {huc_code}: Precision = {precision}, Recall = {recall}, F1 Score = {f1}, IoU = {iou}, Kappa = {kappa}")
            
            # Store the results for this huc_code
            metrics.append({
                'huc_code': huc_code,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'IoU': iou,
                'kappa': kappa
            })
        
        # Log the WandB table
        wandb.log({f"evaluation_metrics_cluster_{scenario['cluster_id']}": table})
        
        # Save the metrics to a CSV file
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(self.args.metrics_save_path, index=False)

        return metrics_df
