from concurrent import futures
import grpc
import numpy as np
import json

import federated_pb2
import federated_pb2_grpc

from .components import ClientManager, ModelRepository, AggregationEngine, Orchestrator

class FederatedLearningServicer(federated_pb2_grpc.FederatedLearningServicer):
    def __init__(self, orchestrator, client_manager, model_repo, config):
        self.orchestrator = orchestrator
        self.client_manager = client_manager
        self.model_repo = model_repo
        self.CLIENTS_PER_ROUND = config['clients_per_round']

    def RegisterClient(self, request, context):
        self.client_manager.register_client(request.client_id)
        
        if not self.orchestrator.is_round_active and self.client_manager.get_available_clients_count() >= self.CLIENTS_PER_ROUND:
            self.orchestrator.start_new_round()
        
        return federated_pb2.ClientRegistrationResponse(message=f"Client {request.client_id} registered.")

    def GetLatestModel(self, request, context):
        round_number, weights = self.model_repo.get_global_model()
        partner_id, seed = self.orchestrator.get_pairing_info_for_client(request.client_id)
        
        status_str = self.orchestrator.training_status
        status_enum = federated_pb2.COMPLETED if status_str == "COMPLETED" else federated_pb2.IN_PROGRESS
        
        return federated_pb2.GetLatestModelResponse(
            round_number=round_number, 
            weights=weights.tobytes(), 
            partner_id=partner_id, 
            shared_seed=seed,
            status=status_enum
        )

    def SubmitModelUpdate(self, request, context):
        update = np.frombuffer(request.weights, dtype=np.float32)
        self.orchestrator.receive_client_update(request.client_id, request.round_number, update)
        return federated_pb2.ModelUpdateResponse(message="Update received.")
    
    def SubmitEvaluationResult(self, request, context):
        self.orchestrator.receive_evaluation_result(request.client_id, request.round_number, request.loss, request.metric)
        return federated_pb2.ModelUpdateResponse(message="Evaluation received.")

def serve():
    with open('config.json', 'r') as f:
        config = json.load(f)

    client_manager = ClientManager(config['clients_per_round'])
    model_repo = ModelRepository(config)
    aggregator = AggregationEngine()
    orchestrator = Orchestrator(client_manager, model_repo, aggregator, config)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_pb2_grpc.add_FederatedLearningServicer_to_server(
        FederatedLearningServicer(orchestrator, client_manager, model_repo, config),
        server
    )
    server.add_insecure_port('[::]:50051')
    print("gRPC server started on port 50051.")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()