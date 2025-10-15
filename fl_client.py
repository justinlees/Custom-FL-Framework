import numpy as np
import time
import argparse
import json
import grpc
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

import federated_pb2
import federated_pb2_grpc

SERVER_ADDRESS = 'localhost:50051'

# --- 1. Auto-detect and set the device (GPU or CPU) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    # ... (Net class is unchanged)
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)
    def forward(self, x):
        out = self.layer1(x); out = self.layer2(out)
        out = out.reshape(out.size(0), -1); out = self.fc(out)
        return out

class FLClient:
    def __init__(self, client_id: str, data_path: str):
        self.client_id = client_id
        self.data_path = data_path
        self.current_round = -1

        self.channel = grpc.insecure_channel(SERVER_ADDRESS)
        self.stub = federated_pb2_grpc.FederatedLearningStub(self.channel)

        with open('config.json', 'r') as f: config = json.load(f)
        self.local_epochs = config['training_params']['local_epochs']
        self.learning_rate = config['training_params']['learning_rate']
        self.batch_size = config['training_params']['batch_size']
        self.grad_clip_norm = config['training_params']['grad_clip_norm']
        self.clipping_bound = config['privacy_params']['clipping_bound']
        self.noise_multiplier = config['privacy_params']['noise_multiplier']
        
        # --- 2. Move the model to the selected device ---
        self.model = Net(num_classes=config['model_architecture']['num_classes']).to(DEVICE)
        
        self.train_loader, self.test_loader = self.load_data()
        print(f"Client {self.client_id} initialized. Using device: {DEVICE}")

    def load_data(self):
        # ... (This function is unchanged)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
        data = np.load(self.data_path)
        if self.client_id == 'hospital-A':
            train_images, train_labels = data['train_images'][:40000], data['train_labels'][:40000]
            test_images, test_labels = data['test_images'][:3500], data['test_labels'][:3500]
        else:
            train_images, train_labels = data['train_images'][40000:], data['train_labels'][40000:]
            test_images, test_labels = data['test_images'][3500:], data['test_labels'][3500:]
        train_images_tensor = torch.stack([transform(img) for img in train_images])
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).squeeze()
        test_images_tensor = torch.stack([transform(img) for img in test_images])
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long).squeeze()
        train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
        test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def pack_weights(self, model):
        return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

    def unpack_weights(self, model, weights: np.ndarray):
        state_dict = OrderedDict()
        start = 0
        for name, param in model.named_parameters():
            end = start + param.numel()
            state_dict[name] = torch.from_numpy(weights[start:end].reshape(param.shape))
            start = end
        model.load_state_dict(state_dict)

    def evaluate_model(self):
        self.model.eval()
        correct, total, loss = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in self.test_loader:
                # --- 3. Move data to the device for evaluation ---
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = loss / len(self.test_loader) if len(self.test_loader) > 0 else 0
        return avg_loss, accuracy

    def perform_local_training(self, initial_weights_np: np.ndarray):
        print(f"Performing local training for {self.local_epochs} epochs...")
        # --- 4. Move the training model to the device ---
        training_model = Net(num_classes=self.model.fc.out_features).to(DEVICE)
        self.unpack_weights(training_model, initial_weights_np)
        training_model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(training_model.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.local_epochs):
            for images, labels in self.train_loader:
                # --- 5. Move data to the device for training ---
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = training_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(training_model.parameters(), self.grad_clip_norm)
                optimizer.step()
        
        print("Local training complete.")
        return self.pack_weights(training_model)

    # ... (All communication methods and the run loop are unchanged)
    def register_with_server(self):
        try:
            req = federated_pb2.ClientRegistrationRequest(client_id=self.client_id)
            self.stub.RegisterClient(req)
            print(f"Client {self.client_id} registered successfully.")
            return True
        except grpc.RpcError: return False

    def fetch_latest_model(self):
        try:
            req = federated_pb2.GetLatestModelRequest(client_id=self.client_id)
            res = self.stub.GetLatestModel(req)
            if res.status == federated_pb2.COMPLETED:
                print("\n--- FEDERATED TRAINING COMPLETE ---")
                final_weights = np.frombuffer(res.weights, dtype=np.float32).copy()
                self.unpack_weights(self.model, final_weights)
                np.save(f"final_model_{self.client_id}.npy", final_weights)
                print(f"Final model saved locally to 'final_model_{self.client_id}.npy'")
                return "TRAINING_COMPLETE"
            if res.round_number > self.current_round:
                print(f"\nNew model for round {res.round_number} received.")
                self.current_round = res.round_number
                weights = np.frombuffer(res.weights, dtype=np.float32).copy()
                self.unpack_weights(self.model, weights)
                self.round_info = {"initial_weights": weights, "partner_id": res.partner_id, "shared_seed": res.shared_seed}
                return True
            return False
        except grpc.RpcError: return False

    def submit_model_update(self, update: np.ndarray):
        norm = np.linalg.norm(update)
        if norm > self.clipping_bound: update = update * (self.clipping_bound / norm)
        if self.noise_multiplier > 0.0:
            noise = np.random.normal(0, self.clipping_bound * self.noise_multiplier, size=update.shape).astype(np.float32)
            private_update = update + noise
        else:
            private_update = update
        final_update = private_update
        if self.round_info["partner_id"]:
            print(f"Applying SecAgg mask with partner {self.round_info['partner_id']}.")
            rng = np.random.RandomState(self.round_info["shared_seed"])
            mask = rng.randn(len(final_update)).astype(np.float32)
            if self.client_id < self.round_info["partner_id"]: final_update += mask
            else: final_update -= mask
        try:
            req = federated_pb2.ModelUpdateRequest(client_id=self.client_id, round_number=self.current_round, weights=final_update.tobytes())
            self.stub.SubmitModelUpdate(req)
            print(f"Update for round {self.current_round} sent.")
        except grpc.RpcError: pass

    def submit_evaluation_result(self, loss: float, metric: float):
        try:
            req = federated_pb2.EvaluationResult(client_id=self.client_id, round_number=self.current_round, loss=loss, metric=metric)
            self.stub.SubmitEvaluationResult(req)
        except grpc.RpcError: pass

    def run(self):
        if not self.register_with_server(): return
        while True:
            status = self.fetch_latest_model()
            if status == "TRAINING_COMPLETE":
                print("Client shutting down.")
                break
            if status == True:
                loss, acc = self.evaluate_model()
                print(f"  - Evaluated global model (Round {self.current_round}): Loss = {loss:.4f}, Accuracy = {acc:.4f}")
                self.submit_evaluation_result(loss, acc)
                # The subtraction is now done inside submit_model_update logic
                trained_weights = self.perform_local_training(self.round_info["initial_weights"])
                update = trained_weights - self.round_info["initial_weights"]
                self.submit_model_update(update)
            print("Waiting for the next round...")
            time.sleep(5)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Client (gRPC)")
    parser.add_argument("--client-id", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    args = parser.parse_args()
    
    client = FLClient(client_id=args.client_id, data_path=args.data_path)
    client.run()