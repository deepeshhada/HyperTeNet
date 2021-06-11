from models.hypertenet import Tenet_Gnn_Seq


class Models(object):
    def __init__(self, params, device='cuda:0'):
        self.model = Tenet_Gnn_Seq(params, device)

    def get_model(self):
        return self.model
