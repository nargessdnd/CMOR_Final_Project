import json
import os

# Load the notebook
notebook_path = 'ESG_Financial_Analysis.ipynb'
try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Create a new code cell
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test cell to verify access and functionality\n",
            "print(\"New cell added successfully!\")\n",
            "import os\n",
            "print(f\"Current working directory: {os.getcwd()}\")"
        ]
    }
    
    # Add the new cell to the notebook
    notebook['cells'].append(new_cell)
    
    # Save the modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Successfully added a new cell to {notebook_path}")
except Exception as e:
    print(f"Error modifying notebook: {e}") 