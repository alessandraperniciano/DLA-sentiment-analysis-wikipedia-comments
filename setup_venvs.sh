#venv for BERT
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        python3 -m venv .venvBERT
        source .venvBERT/bin/activate
        pip3 install -r requirementsBERT.txt
elif [[ "$OSTYPE" == "darwin"* ]]; then
        python3 -m venv .venvBERT
        source .venvBERT/bin/activate
        pip3 install -r requirementsBERT.txt
elif [[ "$OSTYPE" == "msys" ]]; then
        python -m venv .venvBERT
        source .venvBERT/Scripts/activate
        pip install -r requirementsBERT.txt

fi

deactivate


#venv for NN
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        python3 -m venv .venvNN
        source .venvNN/bin/activate
        pip3 install -r requirementsNN.txt
elif [[ "$OSTYPE" == "darwin"* ]]; then
        python3 -m venv .venvNN
        source .venvNN/bin/activate
        pip3 install -r requirementsNN.txt
elif [[ "$OSTYPE" == "msys" ]]; then
        python -m venv .venvNN
        source .venvNN/Scripts/activate
        pip install -r requirementsNN.txt
fi

deactivate