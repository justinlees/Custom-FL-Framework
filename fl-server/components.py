import numpy as np
import random
from typing import List, Dict, Tuple
import json
import time
import sys

class ClientManager:
    def __init__(self, clients_per_round):
        self.clients = {}
        self.selected_clients_for_round: List[str] = []
        self.CLIENTS_PER_ROUND = clients_per_round

    def register_client(self, client_id: str):
        print(f"Registering client: {client_id}")
        self.clients[client_id] = "available"

    def get_available_clients_count(self) -> int:
        return len([status for status in self.clients.values() if status == 'available'])

    def select_clients(self) -> List[str]:
        available_clients = [cid for cid, status in self.clients.items() if status == 'available']
        count = min(self.CLIENTS_PER_ROUND, len(available_clients))
        if count < self.CLIENTS_PER_ROUND: return []
        if count % 2 != 0: count -= 1
        if count == 0: return []

        self.selected_clients_for_round = random.sample(available_clients, count)
        for cid in self.selected_clients_for_round:
            self.clients[cid] = "busy"
        return self.selected_clients_for_round
    
    def make_clients_available(self):
        for cid in self.selected_clients_for_round:
            if cid in self.clients: self.clients[cid] = "available"
        self.selected_clients_for_round = []

class ModelRepository:
    def __init__(self, config):
        model_size = config['model_architecture']['total_weights_count']
        self.global_model_weights = np.random.randn(model_size).astype(np.float32)
        self.current_round = 0
        print(f"ModelRepository initialized with model size: {model_size}")

    def get_global_model(self) -> Tuple[int, np.ndarray]:
        return self.current_round, self.global_model_weights

    def update_global_model(self, new_weights: np.ndarray):
        self.global_model_weights = new_weights
        self.current_round += 1

class AggregationEngine:
    def aggregate_updates(self, current_weights: np.ndarray, updates: List[np.ndarray]) -> np.ndarray:
        if not updates: return current_weights
        print(f"Aggregating {len(updates)} updates...")
        average_update = np.mean(updates, axis=0)
        return current_weights + average_update

class Orchestrator:
    def __init__(self, client_manager, model_repo, aggregator, config):
        self.client_manager = client_manager
        self.model_repo = model_repo
        self.aggregator = aggregator
        self.round_updates: Dict[int, List] = {}
        self.round_evaluations: Dict[int, List] = {}
        self.round_metrics: Dict[int, List] = {}
        self.round_pairings: Dict[str, Tuple[str, int]] = {}
        self.round_start_time = None
        self.is_round_active = False
        self.TOTAL_ROUNDS = config['total_rounds']
        self.CLIENTS_PER_ROUND = config['clients_per_round']
        self.training_status = "IN_PROGRESS"
        print(f"Training will run for a total of {self.TOTAL_ROUNDS} rounds.")

    def start_new_round(self):
        selected_clients = self.client_manager.select_clients()
        if not selected_clients: return

        self.is_round_active = True
        self.round_pairings.clear()
        shuffled_clients = random.sample(selected_clients, len(selected_clients))
        for i in range(0, len(shuffled_clients), 2):
            if i + 1 < len(shuffled_clients):
                c1, c2 = shuffled_clients[i], shuffled_clients[i+1]
                seed = random.randint(0, 2**32 - 1)
                self.round_pairings[c1] = (c2, seed)
                self.round_pairings[c2] = (c1, seed)
        
        current_round, _ = self.model_repo.get_global_model()
        print(f"\n{'='*15} Starting Round {current_round} {'='*15}")
        print(f"Selected clients for this round: {selected_clients}")
        self.round_start_time = time.time()

    def get_pairing_info_for_client(self, client_id: str) -> Tuple[str, int]:
        return self.round_pairings.get(client_id, ("", 0))

    def receive_client_update(self, client_id: str, round_number: int, update_weights: np.ndarray):
        if round_number != self.model_repo.current_round: return
        
        updates = self.round_updates.setdefault(round_number, [])
        updates.append(update_weights)
        print(f"Received update from {client_id} ({len(updates)}/{len(self.client_manager.selected_clients_for_round)})")

        if len(updates) >= len(self.client_manager.selected_clients_for_round):
            self.end_round()

    def receive_evaluation_result(self, client_id: str, round_number: int, loss: float, metric: float):
        self.round_evaluations.setdefault(round_number, []).append(loss)
        self.round_metrics.setdefault(round_number, []).append(metric)
        print(f"Received evaluation from {client_id}: Loss={loss:.4f}, Accuracy={metric:.4f}")

    def end_round(self):
        current_round, current_weights = self.model_repo.get_global_model()
        duration = time.time() - self.round_start_time
        print(f"\n--- Round {current_round} Summary ---")

        if self.round_evaluations.get(current_round):
            avg_loss = np.mean(self.round_evaluations[current_round])
            avg_metric = np.mean(self.round_metrics[current_round])
            print(f"  - Avg. Loss (CrossEntropy): {avg_loss:.4f}")
            print(f"  - Avg. Metric (Accuracy): {avg_metric:.4f}")
        
        new_weights = self.aggregator.aggregate_updates(current_weights, self.round_updates.get(current_round, []))
        self.model_repo.update_global_model(new_weights)
        print("  - Global model updated.")
        print(f"  - Round Duration: {duration:.2f} seconds")
        print(f"{'='*40}")

        self.is_round_active = False
        self.client_manager.make_clients_available()
        self.round_updates.pop(current_round, None)
        self.round_evaluations.pop(current_round, None)
        self.round_metrics.pop(current_round, None)

        if self.model_repo.current_round >= self.TOTAL_ROUNDS:
            print("\n--- FEDERATED TRAINING COMPLETE ---")
            print(f"Reached {self.model_repo.current_round}/{self.TOTAL_ROUNDS} rounds. Final model is ready for download.")
            np.save("final_global_model.npy", self.model_repo.global_model_weights)
            print("Final global model saved to 'final_global_model.npy'")
            self.training_status = "COMPLETED"
        else:
            self.start_new_round()