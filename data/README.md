# Data Directory

The `data/` directory is the central hub for all datasets used in the **Dynamic Neural Network Refinement** project. Proper data management ensures the integrity, reproducibility, and scalability of experiments and model training processes.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Data Management](#data-management)
  - [Using DVC](#using-dvc)
  - [Adding New Datasets](#adding-new-datasets)
  - [Fetching Data](#fetching-data)
- [Data Formats](#data-formats)
- [Best Practices](#best-practices)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The `data/` directory organizes all raw and processed datasets required for training, evaluating, and testing the neural network models. Utilizing [Data Version Control (DVC)](https://dvc.org/) ensures efficient data management, tracking changes, and collaboration across different environments.

## Directory Structure

```

data/ ├── raw/ │ ├── train_dataset.csv │ ├── val_dataset.csv │ └── calibration_dataset.csv ├── processed/ │ ├── train_processed.csv │ ├── val_processed.csv │ └── calibration_processed.csv └── README.md

````

- **raw/**: Contains the original, unprocessed datasets.
- **processed/**: Stores datasets after preprocessing steps, ready for model consumption.

## Data Management

### Using DVC

DVC is integrated into the project to handle large datasets and ensure version control.

#### Initialize DVC

If not already initialized, set up DVC in the project:

```bash
cd data/
dvc init
````

#### Add Remote Storage

Configure remote storage for DVC to store large data files. For example, using AWS S3:

```bash
dvc remote add -d myremote s3://your-bucket/path
```

Ensure that your AWS credentials are correctly set in the `.env` file.

### Adding New Datasets

To add a new dataset to the project:

1. **Place the Dataset:**
    
    Save the dataset in the `raw/` directory.
    
2. **Track with DVC:**
    
    ```bash
    dvc add raw/new_dataset.csv
    ```
    
3. **Commit Changes:**
    
    ```bash
    git add raw/new_dataset.csv.dvc .gitignore
    git commit -m "chore: add new_dataset.csv to DVC tracking"
    ```
    
4. **Push to Remote Storage:**
    
    ```bash
    dvc push
    ```
    

### Fetching Data

To retrieve the datasets from remote storage:

```bash
dvc pull
```

This command downloads all tracked data files to the `data/` directory.

## Data Formats

- **CSV Files:**  
    Primary data format used for storing datasets. Ensure consistency in delimiter usage and encoding.
    
- **Other Formats:**  
    Depending on requirements, other formats like JSON, Parquet, or images may be used. Update preprocessing scripts accordingly.
    

## Best Practices

- **Consistent Naming:**  
    Use clear and descriptive names for datasets to easily identify their purpose and version.
    
- **Data Privacy:**  
    Ensure that sensitive information is handled securely. Avoid committing sensitive data to version control. Use DVC's filtering capabilities if necessary.
    
- **Regular Updates:**  
    Keep datasets up-to-date with the latest information. Use DVC to track changes and maintain historical versions.
    
- **Documentation:**  
    Document the source, purpose, and any preprocessing steps applied to each dataset. This facilitates reproducibility and collaboration.
    

## Contributing

Contributions to the `data/` directory are welcome! To contribute:

1. **Fork the Repository**
    
2. **Create a Feature Branch**
    
    ```bash
    git checkout -b feature/add-new-dataset
    ```
    
3. **Add and Track the Dataset with DVC**
    
    ```bash
    dvc add raw/new_dataset.csv
    ```
    
4. **Commit Your Changes**
    
    ```bash
    git commit -m "chore: add new_dataset.csv to DVC tracking"
    ```
    
5. **Push to Your Fork**
    
    ```bash
    git push origin feature/add-new-dataset
    ```
    
6. **Open a Pull Request**
    
    Provide a clear description of the dataset added and its relevance to the project.
    

For detailed guidelines, refer to the [Best Practices](https://chatgpt.com/docs/best_practices.md) documentation.

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPLv3)](https://chatgpt.com/LICENSE).

## Contact

For questions, suggestions, or support, please open an issue on the [GitHub repository](https://github.com/redx94/Dynamic-Neural-Network-Refinement/issues) or contact the maintainer at [qtt@null.net](mailto:qtt@null.net).

---
