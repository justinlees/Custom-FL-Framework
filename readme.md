# Commands to run the framework

## 1.In the root directory, run the following commands

```
python -m venv venv

(on Windows)
venv\Scripts\activate

(on Mac or Linux)
source venv/bin/activate

(installs dependencies)
pip install -r requirements.txt

(runs gRPC)
python -m grpc_tools.protoc -I./protos --python_out=. --grpc_python_out=. ./protos/federated.proto
```

### Install pathmnist.npz dataset and store it in the root folder.

## 2.Open multiple terminals

### NOTE: Make sure to open every terminal in venv

```
One for server:
python -m fl-server.server

One for hospitalA:
python fl_client.py --client-id="hospital-A" --data-path="pathmnist.npz"

One for hospitalB:
python fl_client.py --client-id="hospital-B" --data-path="pathmnist.npz"

```

## 3.To stop learning, press ctrl+c on every terminal.
