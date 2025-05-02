<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="readmeai/assets/logos/purple.svg" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# MINI_PROJ2

<em></em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/savabs/MINI_PROJ2?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
<img src="https://img.shields.io/github/last-commit/savabs/MINI_PROJ2?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/savabs/MINI_PROJ2?style=default&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/savabs/MINI_PROJ2?style=default&color=0080ff" alt="repo-language-count">

<!-- default option, no dependency badges. -->


<!-- default option, no dependency badges. -->

</div>
<br>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview



---

## Features

|      | Component       | Details                              |
| :--- | :-------------- | :----------------------------------- |
| ⚙️  | **Architecture**  | <ul><li>Multiple model architectures (`model_architecture_1.json` to `model_architecture_4.json`)</li><li>Likely a machine learning project with training configurations</li><li>Best model saved as `best_model.pth`</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Python-based project</li><li>JSON used for configuration files</li><li>Shell scripts present (`.sh` files)</li></ul> |
| 📄 | **Documentation** | <ul><li>Multiple training logs (`training_log_*.txt`)</li><li>Error logging implemented (`error_log.txt`)</li><li>No explicit README or documentation files visible</li></ul> |
| 🔌 | **Integrations**  | <ul><li>No clear external integrations visible</li><li>Possible local data processing pipeline</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Separate files for model architecture and training configs</li><li>Multiple iterations of configurations suggest iterative development</li></ul> |
| 🧪 | **Testing**       | <ul><li>No explicit test files visible</li><li>Training logs may serve as informal validation</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Training logs may contain performance metrics</li><li>`best_model.pth` suggests model optimization</li></ul> |
| 🛡️ | **Security**      | <ul><li>No obvious security measures visible</li><li>Potential sensitive data in training logs</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Python-based, likely using ML libraries (e.g., PyTorch due to `.pth` file)</li><li>JSON parsing required</li></ul> |

---

## Project Structure

```sh
└── MINI_PROJ2/
    └── LLVM_IR_Modeling
        ├── Classifier_poj104
        └── Data_things
```

### Project Index

<details open>
	<summary><b><code>MINI_PROJ2/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
			</table>
		</blockquote>
	</details>
	<!-- LLVM_IR_Modeling Submodule -->
	<details>
		<summary><b>LLVM_IR_Modeling</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ LLVM_IR_Modeling</b></code>
			<!-- Classifier_poj104 Submodule -->
			<details>
				<summary><b>Classifier_poj104</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ LLVM_IR_Modeling.Classifier_poj104</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/training_Simple_MLP.py'>training_Simple_MLP.py</a></b></td>
							<td style='padding: 8px;'>- Implements a multiclass classifier training pipeline using PyTorch<br>- Handles data loading, preprocessing, model architecture definition, and training with early stopping and checkpointing<br>- Utilizes GPU acceleration, gradient scaling, and mixed precision training<br>- Includes logging functionality and configuration management through command-line arguments<br>- Saves model architecture and training configuration for reproducibility and future reference.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/text.txt'>text.txt</a></b></td>
							<td style='padding: 8px;'>- Logs output from a machine learning model training process for graph classification<br>- Records the number of loaded graphs, specifies the use of a DeepGTAT model with cross attention, and indicates the checkpoint save location<br>- Serves as a progress tracker and configuration record for the LLVM IR modeling project, particularly within the Classifier_poj104 component.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/training_Seq2Seq.py'>training_Seq2Seq.py</a></b></td>
							<td style='padding: 8px;'>- Setting up the model architecture and hyperparameters using the <code>ModelArgs</code> dataclass.2<br>- Implementing the training loop, including data loading, model optimization, and evaluation.3<br>- Utilizing PyTorch for deep learning operations and leveraging GPU acceleration where available.4<br>- Incorporating advanced training techniques such as learning rate scheduling and gradient scaling.This file plays a central role in the projects machine learning pipeline, bridging the gap between raw LLVM IR data and the classification output<br>- It's designed to work in conjunction with other components of the project to process and analyze LLVM Intermediate Representation code effectively.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/error_log.txt'>error_log.txt</a></b></td>
							<td style='padding: 8px;'>- Record and document warnings or errors that occur during the training process of the Graph Neural Network (GNN).2<br>- Provide developers with information about potential security risks or future deprecations in the code, particularly related to the use of torch.load function.3<br>- Serve as a debugging tool to help identify and resolve issues in the GNN training process.4<br>- Act as a reference for future improvements or updates to the codebase, especially regarding security considerations and keeping up with library changes.This error log is an important component for maintaining and improving the project, as it helps developers stay informed about potential issues and upcoming changes in the libraries used, particularly in the context of machine learning model training and security.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/training_side.py'>training_side.py</a></b></td>
							<td style='padding: 8px;'>- Implements the training pipeline for a multiclass classifier in the LLVM_IR_Modeling project<br>- Loads preprocessed data, defines a neural network model, sets up training parameters, and executes the training loop<br>- Includes functionality for logging training progress, evaluating on validation data, and periodically saving model checkpoints<br>- Serves as the core training script for the Classifier_poj104 component, preparing the model for subsequent use in the project.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/GTAT_iteration1_512_DH_0.0005_epoch40.txt'>GTAT_iteration1_512_DH_0.0005_epoch40.txt</a></b></td>
							<td style='padding: 8px;'>- Logs training progress for a DeepGTAT model with cross attention on a dataset of 41,678 graphs<br>- Records epoch-by-epoch performance metrics including training loss, training accuracy, validation loss, and validation accuracy<br>- Tracks best model checkpoints based on validation accuracy improvements<br>- Demonstrates the models learning progression over 40 epochs, showing significant accuracy gains and loss reductions in both training and validation sets.</td>
						</tr>
					</table>
					<!-- GNN Submodule -->
					<details>
						<summary><b>GNN</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ LLVM_IR_Modeling.Classifier_poj104.GNN</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/GNN/training_log_2025-03-25_19-49-04.txt'>training_log_2025-03-25_19-49-04.txt</a></b></td>
									<td style='padding: 8px;'>- Its part of a larger project involving LLVM IR modeling and classification.2<br>- The log is timestamped, allowing for easy tracking of different training runs.3<br>- It captures warnings, such as the FutureWarning about <code>torch.load</code>, which can help developers address potential issues or deprecations in future versions of the libraries used.This log file is essential for:-Tracking the training progress over time-Identifying any issues or warnings that arise during training-Providing a historical record of the training process for future reference or comparisonIn the context of the project architecture, this log file serves as an output artifact of the GNN training process, likely generated by the GNN1_training.py script<br>- Its a valuable resource for data scientists and machine learning engineers working on this LLVM IR modeling project to assess the performance and behavior of their GNN classifier.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/GNN/GNN_training.py'>GNN_training.py</a></b></td>
									<td style='padding: 8px;'>- Setting up the training environment with necessary imports and hyperparameters.2<br>- Defining the GNN model architecture, focusing on the enhanced GTAT layer.3<br>- Implementing the training loop, including data loading, model optimization, and evaluation.This file plays a central role in the project's machine learning pipeline, bridging the gap between the LLVM IR graph representations and the classification task<br>- It's designed to work with the graph data structures created elsewhere in the project, likely processing the output of LLVM IR parsing and graph construction steps.The model trained here is expected to be used for classifying different aspects of LLVM IR code, contributing to the overall goal of the LLVM_IR_Modeling project in analyzing and understanding LLVM Intermediate Representation.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/GNN/training_log_2025-03-23_12-25-04.txt'>training_log_2025-03-23_12-25-04.txt</a></b></td>
									<td style='padding: 8px;'>- Monitoring the training process2<br>- Debugging issues that may arise3<br>- Analyzing model performance over time4<br>- Keeping a historical record of training runsThe log file is automatically generated and updated as the training progresses, capturing important information that can help developers and researchers understand the model's behavior and make improvements.In the context of the entire codebase architecture, this log file is a crucial part of the model development and evaluation pipeline, providing insights into the training process and helping to ensure the quality and reliability of the GNN classifier for the LLVM IR Modeling project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/GNN/training_log_2025-03-23_14-14-44.txt'>training_log_2025-03-23_14-14-44.txt</a></b></td>
									<td style='padding: 8px;'>- Training progress2<br>- Model performance metrics3<br>- Potential warnings or errors encountered during the training processThis log file serves as a crucial resource for:-Monitoring the training process-Debugging issues that may arise-Analyzing model performance over time-Keeping a record of different training runs for comparisonThe log file is automatically generated and updated during the training of the GNN model, providing valuable insights into the models behavior and helping researchers or developers track the progress of their experiments within the LLVM IR Modeling project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/GNN/graph_generate.py'>graph_generate.py</a></b></td>
									<td style='padding: 8px;'>- Generates graph representations from CSV files containing node features and edge indices for a machine learning project<br>- Processes data from multiple class folders, creates PyTorch Geometric Data objects for each graph, and assigns class labels<br>- Compiles all graphs into a list and saves the dataset in a format compatible with PyTorch Geometrics InMemoryDataset for further use in graph-based machine learning tasks.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/GNN/training_log_2025-03-24_18-59-08.txt'>training_log_2025-03-24_18-59-08.txt</a></b></td>
									<td style='padding: 8px;'>- Record the training process and progress of the GNN model2<br>- Capture any warnings, errors, or important messages during the training3<br>- Provide a timestamped record of the training session (as indicated by the date in the filename)This log file serves as a crucial component for monitoring, debugging, and analyzing the model's training performance<br>- It allows researchers and developers to review the training process, identify potential issues, and track the model's improvement over time.The log is an output file rather than an active part of the codebase, but it plays an important role in the overall development and refinement of the GNN classifier within the projects architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/GNN/GNN1_training.py'>GNN1_training.py</a></b></td>
									<td style='padding: 8px;'>- Implements a Graph Neural Network (GNN) training pipeline for classifying LLVM IR code<br>- Utilizes an advanced GAT-LSTM architecture, mixed precision training, and gradient clipping for improved performance<br>- Includes data loading, model definition, training loop with progress bars, and evaluation functions<br>- Incorporates best model saving and learning rate scheduling to optimize training outcomes<br>- Designed for the LLVM_IR_Modeling projects Classifier_poj104 component.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/GNN/data_check.py'>data_check.py</a></b></td>
									<td style='padding: 8px;'>- Loads and inspects a processed graph dataset from a specified file path<br>- Analyzes the structure of the loaded data, determining if it contains multiple graphs or a single graph<br>- Prints key information about the dataset, including the number of graphs, nodes, edges, node feature shapes, and graph labels<br>- Provides a quick overview of the datasets characteristics for further analysis or processing.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/GNN/training_log_2025-03-25_00-57-41.txt'>training_log_2025-03-25_00-57-41.txt</a></b></td>
									<td style='padding: 8px;'>- Record training progress and results2<br>- Capture any warnings or issues that occur during the training process3<br>- Provide a timestamped record of the training sessionThis log file is part of a larger project structure that involves LLVM IR modeling and classification, specifically for the POJ-104 dataset<br>- It's likely used for debugging, performance analysis, and tracking the model's training over time.The warning message captured in the log suggests that the code is using an older version of PyTorch's <code>torch.load</code> function, which may need to be updated in future versions of the project.Overall, this file serves as a crucial component for monitoring and analyzing the GNN models training process within the context of the LLVM IR modeling and classification project.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/GNN/script.sh'>script.sh</a></b></td>
									<td style='padding: 8px;'>- Executes GNN1_training.py in the background, redirecting output to a log file<br>- Enables continuous model training without occupying the terminal<br>- Includes a command to list processes running under a specific username, facilitating monitoring and management of running scripts<br>- Essential for automating and tracking the training process in the LLVM_IR_Modeling projects Classifier_poj104 GNN component.</td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- Seq2Seq_transformer Submodule -->
					<details>
						<summary><b>Seq2Seq_transformer</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ LLVM_IR_Modeling.Classifier_poj104.Seq2Seq_transformer</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Seq2Seq_transformer/plot_val_train.py'>plot_val_train.py</a></b></td>
									<td style='padding: 8px;'>- Extracts training and validation metrics from log files, combining data from multiple sources<br>- Generates a comprehensive visualization of training loss, validation loss, and validation accuracy across epochs<br>- The resulting plot is saved as an image file, providing a clear visual representation of model performance trends throughout the training process.</td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- Simple_MLP Submodule -->
					<details>
						<summary><b>Simple_MLP</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ LLVM_IR_Modeling.Classifier_poj104.Simple_MLP</b></code>
							<!-- model_data_2 Submodule -->
							<details>
								<summary><b>model_data_2</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ LLVM_IR_Modeling.Classifier_poj104.Simple_MLP.model_data_2</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_2/model_architecture_2.json'>model_architecture_2.json</a></b></td>
											<td style='padding: 8px;'>- Defines the architecture of a simple multi-layer perceptron (MLP) neural network for classifying LLVM IR code<br>- Specifies input size, number of output classes, and layer structure including linear transformations and ReLU activations<br>- Designed for the Classifier_poj104 project, this model configuration aids in processing and categorizing LLVM intermediate representations within the broader LLVM_IR_Modeling framework.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_2/training_config_2.json'>training_config_2.json</a></b></td>
											<td style='padding: 8px;'>- Configuration settings for training a machine learning model are defined in the training_config_2.json file<br>- It specifies hyperparameters such as batch size, learning rate, and epochs, as well as file paths for data, logs, and model architecture<br>- Multiple configurations are provided, allowing for experimentation with different settings to optimize model performance and training efficiency.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_2/training_log_2.txt'>training_log_2.txt</a></b></td>
											<td style='padding: 8px;'>- Training log records the progress of a machine learning model for the LLVM IR Modeling project<br>- It captures multiple training sessions, documenting epoch-wise performance metrics including train loss, validation loss, and validation accuracy<br>- The log demonstrates model improvements over time, with early stopping mechanisms to prevent overfitting<br>- It also tracks model saving events and resumption of training from previously saved states.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_2/best_model.pth'>best_model.pth</a></b></td>
											<td style='padding: 8px;'>- The project structure (directory layout)2<br>- The specific file you want summarized3<br>- The contents of that file or at least its key functions/featuresIf you could provide this information, Id be happy to analyze it and give you a concise summary focusing on the file's main purpose and its role in the broader project context, without delving into technical implementation details.</td>
										</tr>
									</table>
								</blockquote>
							</details>
							<!-- model_data_1 Submodule -->
							<details>
								<summary><b>model_data_1</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ LLVM_IR_Modeling.Classifier_poj104.Simple_MLP.model_data_1</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_1/model_architecture_1.json'>model_architecture_1.json</a></b></td>
											<td style='padding: 8px;'>- Defines the architecture of a simple Multi-Layer Perceptron (MLP) model for the LLVM IR Modeling projects classifier<br>- Specifies input size, number of output classes, and layer configuration including linear transformations and ReLU activations<br>- Serves as a blueprint for constructing the neural network used in classifying LLVM IR code samples across 104 different categories.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_1/training_config_1.json'>training_config_1.json</a></b></td>
											<td style='padding: 8px;'>- Defines training configuration parameters for a machine learning model in the LLVM_IR_Modeling project<br>- Specifies data directory, batch size, learning rate, epochs, logging details, model storage locations, early stopping patience, and architecture file path<br>- Serves as a central configuration point for the Simple_MLP classifier, enabling consistent and reproducible training runs within the Classifier_poj104 component of the project.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_1/training_log_1.txt'>training_log_1.txt</a></b></td>
											<td style='padding: 8px;'>- Training log for a Simple MLP classifier in the LLVM_IR_Modeling project<br>- Records model performance across multiple epochs, tracking train loss, validation loss, and accuracy<br>- Demonstrates iterative improvement, model saving at best performances, and early stopping to prevent overfitting<br>- Includes two training sessions, with the second resuming from the best saved model, showcasing continued refinement of the classifiers performance.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_1/best_model.pth'>best_model.pth</a></b></td>
											<td style='padding: 8px;'>- The project structure (directory layout)2<br>- The specific file path you want me to focus on3<br>- Ideally, the contents of that file or a description of its functionalityIf you can provide this information, Id be happy to deliver a concise summary highlighting the main purpose and use of the code file within the context of the entire codebase architecture, focusing on what the code achieves rather than technical implementation details.</td>
										</tr>
									</table>
								</blockquote>
							</details>
							<!-- model_data_3 Submodule -->
							<details>
								<summary><b>model_data_3</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ LLVM_IR_Modeling.Classifier_poj104.Simple_MLP.model_data_3</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_3/training_log_3.txt'>training_log_3.txt</a></b></td>
											<td style='padding: 8px;'>- Training log records multiple iterations of a machine learning models training process for the LLVM_IR_Modeling project<br>- It captures epoch-wise performance metrics including train loss, validation loss, and validation accuracy<br>- The log demonstrates the models improvement over time, with periodic model saves and early stopping triggers to prevent overfitting<br>- It also includes timestamps and indicates when training resumes from previously saved models.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_3/training_config_3.json'>training_config_3.json</a></b></td>
											<td style='padding: 8px;'>- Training configuration file for a Simple MLP classifier in the LLVM_IR_Modeling project<br>- Contains multiple JSON objects specifying hyperparameters and file paths for model training<br>- Defines batch sizes, learning rates, epochs, early stopping patience, and locations for data, logs, and model architecture<br>- Enables experimentation with different training settings to optimize the classifiers performance on the POJ-104 dataset.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_3/model_architecture_3.json'>model_architecture_3.json</a></b></td>
											<td style='padding: 8px;'>- Defines the architecture of a multi-layer perceptron (MLP) neural network for classifying LLVM IR code into 104 categories<br>- Specifies an input size of 300 features and outlines five linear layers with ReLU activations, progressively reducing dimensionality from 1024 to 104 output classes<br>- This configuration is crucial for the models structure and performance in the LLVM IR classification task.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_3/best_model.pth'>best_model.pth</a></b></td>
											<td style='padding: 8px;'>- The contents of the specific code file you want summarized2<br>- The project structure or file tree3<br>- Any additional context about the project's purpose or architectureIf you can provide this information, Id be happy to analyze it and deliver a concise summary focusing on the file's main purpose and its role within the larger project architecture.</td>
										</tr>
									</table>
								</blockquote>
							</details>
							<!-- model_data_4 Submodule -->
							<details>
								<summary><b>model_data_4</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ LLVM_IR_Modeling.Classifier_poj104.Simple_MLP.model_data_4</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_4/model_architecture_4.json'>model_architecture_4.json</a></b></td>
											<td style='padding: 8px;'>- Defines the architecture of a multi-layer perceptron (MLP) neural network for classifying LLVM IR code into 104 categories<br>- The model consists of five linear layers with batch normalization, ReLU activation, and dropout between each<br>- Starting with 300 input features, the network progressively reduces dimensionality through hidden layers of 1024, 512, 256, and 128 neurons before outputting 104 class probabilities.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_4/training_config_4.json'>training_config_4.json</a></b></td>
											<td style='padding: 8px;'>- Training configuration for a Simple MLP classifier in the LLVM_IR_Modeling project<br>- Specifies hyperparameters, data paths, and model settings for multiple training runs<br>- Includes batch size, learning rate, epochs, early stopping patience, and file locations for data, logs, and model architecture<br>- Enables consistent and reproducible training across different iterations of the classifier.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_4/best_model.pth'>best_model.pth</a></b></td>
											<td style='padding: 8px;'>- And FILE PA sections<br>- Without this crucial information about the project structure and the specific file you want me to summarize, I'm unable to provide an accurate and meaningful summary.To help you effectively, I would need:1<br>- The complete project structure2<br>- The full file path of the specific code file you want summarized3<br>- Ideally, the contents of that file or at least a description of its roleOnce you provide this information, Ill be able to deliver a succinct summary highlighting the main purpose and use of the code file in relation to the entire codebase architecture, focusing on what the code achieves without delving into technical implementation details.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/Simple_MLP/model_data_4/training_log_4.txt'>training_log_4.txt</a></b></td>
											<td style='padding: 8px;'>- Records training progress for a machine learning model, tracking epochs, losses, and accuracies across multiple training sessions<br>- Logs model saving events, GPU usage, and configuration updates<br>- Demonstrates iterative improvement in model performance, with fluctuations in validation loss and accuracy<br>- Provides a detailed chronological account of the training process, enabling analysis of model behavior and optimization efforts over time.</td>
										</tr>
									</table>
								</blockquote>
							</details>
						</blockquote>
					</details>
					<!-- training_lst_tran Submodule -->
					<details>
						<summary><b>training_lst_tran</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ LLVM_IR_Modeling.Classifier_poj104.training_lst_tran</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Classifier_poj104/training_lst_tran/transfomer_lstm.py'>transfomer_lstm.py</a></b></td>
									<td style='padding: 8px;'>- Implements a hybrid neural network model combining transformer and LSTM architectures for classifying LLVM IR code<br>- Defines custom dataset handling, dynamic batching, and training procedures<br>- Utilizes GPU acceleration and mixed precision training<br>- Includes logging functionality and model checkpointing<br>- Designed for processing instruction embeddings from a dataset of 104 classes, with provisions for training, validation, and model evaluation.</td>
								</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<!-- Data_things Submodule -->
			<details>
				<summary><b>Data_things</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ LLVM_IR_Modeling.Data_things</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Data_things/check_data.py'>check_data.py</a></b></td>
							<td style='padding: 8px;'>- Checks and explores preprocessed training data for a machine learning project<br>- Loads numpy arrays containing features (X_train) and labels (y_train) from a specified directory<br>- Prints dataset information, including shapes and number of classes<br>- Provides a function to examine individual samples, displaying their one-hot encoded labels, class indices, and data visualization when possible<br>- Aids in verifying data integrity and understanding the datasets structure before model training.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Data_things/data_processing.py'>data_processing.py</a></b></td>
							<td style='padding: 8px;'>- Processes and prepares LLVM IR instruction embeddings for machine learning tasks<br>- Loads class data in chunks, pads sequences, creates attention masks, and generates one-hot encoded labels<br>- Splits the dataset into training, validation, and test sets<br>- Saves processed data in batches and final datasets as PyTorch tensors, facilitating efficient loading and training of deep learning models for LLVM IR analysis and classification.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Data_things/gen_prog_embed.py'>gen_prog_embed.py</a></b></td>
							<td style='padding: 8px;'>- Generates program embeddings from LLVM IR files using IR2Vec<br>- Traverses a directory structure, processes.ll files, and creates corresponding instruction embeddings<br>- Saves the resulting embeddings as NumPy arrays in a parallel directory structure<br>- Handles potential errors during processing and ensures output directories exist<br>- Facilitates the conversion of LLVM IR representations into vector embeddings for further analysis or machine learning tasks.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Data_things/count_max_inst.py'>count_max_inst.py</a></b></td>
							<td style='padding: 8px;'>- Analyzes instruction embeddings from LLVM IR files, calculating statistics on instruction list lengths across different program classes<br>- Computes key metrics like maximum, minimum, mean, median, and percentiles<br>- Generates a histogram to visualize the distribution of instruction list lengths<br>- Provides insights into the complexity and size variations of programs in the dataset, aiding in understanding the characteristics of the LLVM IR representations.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Data_things/graph_generate.py'>graph_generate.py</a></b></td>
							<td style='padding: 8px;'>- Defines a GraphDataset class for processing and loading graph data from CSV files<br>- Converts folder structures into graph representations, assigning class labels based on folder names<br>- Extracts node features and edge information, creating PyTorch Geometric Data objects<br>- Implements data loading, processing, and saving functionalities, enabling efficient handling of multiple graphs for machine learning tasks.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Data_things/usingptfiles.py'>usingptfiles.py</a></b></td>
							<td style='padding: 8px;'>- Processes and prepares PyTorch data files for machine learning tasks<br>- Loads variable-length sequence data from.pt files, pads sequences to a uniform length, combines and shuffles the data, then splits it into training, validation, and test sets<br>- Saves the processed datasets as separate files, facilitating efficient data handling for subsequent model training and evaluation in the LLVM IR modeling project.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Data_things/generate_graph.sh'>generate_graph.sh</a></b></td>
							<td style='padding: 8px;'>- Generates graph representations from LLVM IR files using IR2Vec<br>- Processes input files organized by class, creating corresponding output directories<br>- Executes IR2Vec on each.ll file, producing CSV graph outputs<br>- Maintains original file naming conventions and organizes results in a structured manner<br>- Facilitates the conversion of LLVM IR to graph-based representations for further analysis or machine learning tasks.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Data_things/512_Size_constrain.py'>512_Size_constrain.py</a></b></td>
							<td style='padding: 8px;'>- Processes and standardizes instruction vector data for machine learning tasks<br>- Downsamples vectors to a fixed length of 512 using average pooling, ensuring consistent input size<br>- Iterates through program files in the instruction_embeddings directory, applies downsampling, and saves the processed vectors in a new processed_instruction_embeddings directory<br>- Facilitates uniform data preparation for subsequent analysis or model training in the LLVM IR modeling project.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Data_things/checking_pt_files.py'>checking_pt_files.py</a></b></td>
							<td style='padding: 8px;'>- Inspects and analyzes PyTorch (.pt) files within a specified directory<br>- Loads each file, determines its content type (tensor, dictionary, or list), and prints shape information for tensors<br>- Handles various data structures, including nested dictionaries and lists containing tensors<br>- Provides error handling for file processing issues<br>- Useful for quickly assessing the structure and dimensions of saved PyTorch data in a project.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Data_things/playingwithdata.py'>playingwithdata.py</a></b></td>
							<td style='padding: 8px;'>- Processes and manipulates data for the LLVM IR Modeling project<br>- Handles data loading, preprocessing, and transformation tasks essential for preparing input for machine learning models<br>- Implements functions to clean, normalize, and format data, ensuring consistency and compatibility with the projects requirements<br>- Serves as a crucial component in the data pipeline, bridging raw data sources with the modeling and analysis stages of the project.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/savabs/MINI_PROJ2/blob/master/LLVM_IR_Modeling/Data_things/data_processing_instruction.py'>data_processing_instruction.py</a></b></td>
							<td style='padding: 8px;'>- Analyzes and summarizes instruction embedding data stored in.npy files within class-specific directories<br>- Processes each class, counting samples and recording data shapes<br>- Provides a comprehensive overview of the dataset, including class-wise details, total sample count, unique shapes, and the most common shape<br>- Identifies potential inconsistencies in sample shapes, suggesting padding or truncation for uniformity if necessary.</td>
						</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python

### Installation

Build MINI_PROJ2 from the source and intsall dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone https://github.com/savabs/MINI_PROJ2
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd MINI_PROJ2
    ```

3. **Install the dependencies:**

echo 'INSERT-INSTALL-COMMAND-HERE'

### Usage

Run the project with:

echo 'INSERT-RUN-COMMAND-HERE'

### Testing

Mini_proj2 uses the {__test_framework__} test framework. Run the test suite with:

echo 'INSERT-TEST-COMMAND-HERE'

---


## Contributing

- **💬 [Join the Discussions](https://github.com/savabs/MINI_PROJ2/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/savabs/MINI_PROJ2/issues)**: Submit bugs found or log feature requests for the `MINI_PROJ2` project.
- **💡 [Submit Pull Requests](https://github.com/savabs/MINI_PROJ2/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/savabs/MINI_PROJ2
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/savabs/MINI_PROJ2/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=savabs/MINI_PROJ2">
   </a>
</p>
</details>

---

## License

Mini_proj2 is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## Acknowledgments

- Credit `contributors`, `inspiration`, `references`, etc.

<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square


---
