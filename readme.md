# Custom Federated Learning Framework with privacy techniques

In this framework, we currently focus on only PathMNIST data for the Federated Learning. The server is independent of training configurations makes this framework robust. The framework uses gRPC protobufs for communication which helps in less communication overhead for gradient details transfer to the server. But this framework currently runs on local Machine with the help of multiple terminals being the server and clients. You can also run this framework on multiple devices by connecting to same network as the server and a change in the client side file code where the SERVER_ADDRESS value should be changed to your network IPv4 address.

This framework implements the Federated learning task but currently does not support non-iid data. So the data should of similar type across the clients. The final model will be downloaded at the client side system for helping to predict new data faster.

#### Privacy Techniques:

    This Framework also provides privacy by implementing Differential Privacy(DP) and Secure-Agg mechanisms. So the local gradients will be secured against Gradient and Model inversion attacks.

Below, we mentioned the steps to follow for using this framework to start Federated Learning and predict the new unseen data.

#### NOTE:Currently this framework doesn't have a global server. So we have provided the server files as well to run at your system.

## Step-1 : Clone the repo

a. After cloning the repo, in the root directory, create virtual environment:

```
python -m venv FLenv

```

This helps to keep all the projects related libraries intact.

b. After creating the virtual environment, activate it:

```
(on Windows)
FLenv\Scripts\activate

(on Mac or Linux)
source FLenv/bin/activate

```

c. Now install all the libraries required for the framework to run:

```
(installs dependencies)
pip install -r requirements.txt
if nvidia GPU present:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else:
pip install torch torchvision
```

d. Now run the below command for gRPC protobuf working:

```
(runs gRPC)
python -m grpc_tools.protoc -I./protos --python_out=. --grpc_python_out=. ./protos/federated.proto

```

## Step-2: Install PathMNIST dataset

a. You can download the official PathMNIST dataset here : https://zenodo.org/records/10519652/files/pathmnist.npz?download=1 at Zenodo organization. After installation, place this dataset in the root folder.

## Step-3: Start the Federated Learning (localmachine)

a.Run the following command for the required dataset splits according to the no.of clients:

```
python split_data.py
```

### NOTE: We are splitting data to mimic the iid data in real-world scenario. Since the clients doesn't have iid data we are splitting the orginal data for training.

b. Open multiple terminals which acts as multiple edge devices.

#### NOTE: Make sure to open every terminal in venv

One for server:

```
python -m fl-server.server

```

One for hospital1:

```
python fl_client.py --client-id="hospital-1" --data-path="hospital_1_data.npz"

```

One for hospital2:

```
python fl_client.py --client-id="hospital-2" --data-path="hospital_2_data.npz"

```

At the end of the training, the local model for each client will be downloaded and for the server as well. Since we are running all the three in the same localmachine, you will see three final model files.

## Step-4: Start the Federated learning (Same Network)

a. As mentioned in the context, you need to change the SERVER_ADDRESS value to networks IPv4 address in fl_client.py
b. Make sure the devices are running on the same network as server.
c. First run the split_data.py for getting dataset splits according to no.of clients participating.
d. Share the datasets to the clients.

### NOTE: We are splitting data to mimic the iid data in real-world scenario. Since the clients doesn't have iid data we are splitting the orginal data for training.

e. Follow the same commands as in the localmachine step(one client runs hospitalA and other runs hospitalB and server runs server command).

## Step-5: To stop learning, press ctrl+c on server.

For modifying the training parameters, change the parameters in config.json
