# PolicyAI

PolicyAI is an AI-powered tool that leverages GPT-4 to analyze datasets and provide intelligent, query-driven insights. Designed for flexibility, PolicyAI allows users to interact with their data using natural language, making data exploration and policy analysis accessible, efficient, and insightful.

## Features

- **LLM-Driven Insights:** Uses OpenAI's GPT-4 to parse and interpret complex datasets.
- **Flexible Querying:** Input queries in plain language to get customized outputs.
- **Dataset Agnostic:** Works with a variety of structured data formats.
- **Actionable Outputs:** Obtain summaries, recommendations, or data extractions based on your questions.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API Key (for GPT-4 access)
- Required dependencies (see `requirements.txt`)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Karthic-45/Policy_Ai.git
cd Policy_Ai
pip install -r requirements.txt
```

### Configuration

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=your_openai_api_key
```

### Usage

1. Prepare your dataset (CSV, JSON, or supported format).
2. Run the main application:

```bash
python main.py --dataset path/to/your/data.csv --query "Summarize the key trends in policy adoption."
```

3. View the output in your terminal or as a saved file, depending on your configuration.

## Example Queries

- "What are the main findings from this policy dataset?"
- "Extract all policies related to healthcare from 2020-2024."
- "Summarize the differences between these two policy documents."

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenAI](https://openai.com/) for the GPT-4 API
- All contributors

---

Feel free to reach out or open an issue if you have questions or suggestions!
