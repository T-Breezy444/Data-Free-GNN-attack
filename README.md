# Overview

This repository implements and recreates the attacks described in the
paper **\"Unveiling the Secrets without Data: Can Graph Neural Networks
Be Exploited through Data-Free Model Extraction Attacks?\"**, presented
at the 33rd USENIX Security Symposium (2024). The implementation
demonstrates data-free model extraction attacks on Graph Neural Networks
(GNNs) without access to actual graph data or node features.

# Architecture

The project is structured as follows:

    /stealgnn (root directory)
        main.py
        /models
            __init__.py
            victim.py
            generator.py
            surrogate.py
        /attacks
            __init__.py
            attack1.py
            attack2.py
            attack3.py
        requirements.txt

## Key Components

### main.py

The entry point for running experiments. It handles dataset loading,
model initialization, attack execution, and result reporting.

### models/

-   `victim.py`: Implements the victim GNN models for different datasets
    (Cora, Computers, Pubmed, OGB-Arxiv).

-   `generator.py`: Contains the GraphGenerator class for creating
    synthetic graphs.

-   `surrogate.py`: Implements the surrogate model that attempts to
    mimic the victim model.

### attacks/

-   `attack1.py`: Implements the Type I attack.

-   `attack2.py`: Implements the Type II attack.

-   `attack3.py`: Implements the Type III attack.

# Attack Types

-   **Type I Attack**: Uses gradients from both surrogate and estimated
    victim model.

-   **Type II Attack**: Uses gradients only from the surrogate model.

-   **Type III Attack**: Uses two surrogate models to capture more
    complex knowledge.

# Datasets

The implementation supports four datasets:

-   Cora

-   Computers

-   Pubmed

-   OGB-Arxiv

# How to Run

1.  Install dependencies:

            pip install -r requirements.txt

2.  Run an attack:

            python main.py <attack_type> <dataset_name>

    Where `<attack_type>` is 1, 2, or 3, and `<dataset_name>` is 'cora',
    'computers', 'pubmed', or 'ogb-arxiv'.

# Evaluation Metrics

-   Accuracy

-   Fidelity

-   F1 Score

-   Confusion Matrix

# Output

The script generates:

-   Printed statistics

-   Confusion matrix plot

-   Loss plot

-   PDF report with detailed results

# License

This project is licensed under the MIT License. See the `LICENSE` file
in the repository for the full license text.

# References

Zhuang, Y., Shi, C., Zhang, M., Chen, J., Lyu, L., Zhou, P., & Sun, L.
(2024). *Unveiling the Secrets without Data: Can Graph Neural Networks
Be Exploited through Data-Free Model Extraction Attacks?* USENIX
Security Symposium, 2024.
<https://www.usenix.org/conference/usenixsecurity24/presentation/zhuang>
