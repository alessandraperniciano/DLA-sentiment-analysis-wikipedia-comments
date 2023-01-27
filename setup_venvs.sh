python3 -m venv .venvBERT

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        source .venvBERT/bin/activate
elif [[ "$OSTYPE" == "darwin"* ]]; then
        source .venvBERT/bin/activate
elif [[ "$OSTYPE" == "msys" ]]; then
        source .venvBERT/Scripts/activate
fi

pip3 install -r requirementsBERT.txt

deactivate

python3 -m venv .venvNN

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        source .venvNN/bin/activate
elif [[ "$OSTYPE" == "darwin"* ]]; then
        source .venvNN/bin/activate
elif [[ "$OSTYPE" == "msys" ]]; then
        source .venvNN/Scripts/activate
fi

pip3 install -r requirementsNN.txt

deactivate