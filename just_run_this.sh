# Download the llamafile
wget https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q4_0.llamafile

# give it permission to execute
chmod +x mistral-7b-instruct-v0.2.Q4_0.llamafile

# Make the storage directory
mkdir storage

# Start the llamafile server
./mistral-7b-instruct-v0.2.Q4_0.llamafile --server --nobrowser --embedding --port 8080

# Run the py script
python3 idec8003forge.py

