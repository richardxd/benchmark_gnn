import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer, GatedGCNLayerEdgeFeatOnly, GatedGCNLayerIsotropic
from layers.mlp_readout_layer import MLPReadout

class GatedGCNNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.n_classes = n_classes
        self.device = net_params['device']
        
        self.layer_type = {
            "edgereprfeat": GatedGCNLayer,
            "edgefeat": GatedGCNLayerEdgeFeatOnly,
            "isotropic": GatedGCNLayerIsotropic,
        }.get(net_params['layer_type'], GatedGCNLayer)
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        self.layers = nn.ModuleList([ self.layer_type(hidden_dim, hidden_dim, dropout,
                                                      self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(self.layer_type(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(2*out_dim, n_classes)
    
    def compute_tour_len(self, y_pred_edge, x_edges_values):
        # torch.argmax(g.edata['e']) 
        y = F.softmax(y_pred_edge, dim=1)
        y = y.argmax(dim=1).squeeze()
        # print(edge_feat[0])
        # print("edge pred:", y_pred_edge[0])
        # for i in range(len(edge_labels)):
        #     print("before boolean value ambiguous:", edge_labels[i])
        #     if edge_labels[i] == 1:
        #         tour_len += edge_feat[i]
        # print(x_edges_values.shape)
        # print(y.shape)
        # print(x_edges_values[0:100])
        # print(sum(y))
        result = y.float() @ x_edges_values.float()
        tour_len = (y.float() @ x_edges_values.float()) / 2
        # print("the predicted tour length is therefore:", tour_len)
        return tour_len


    def forward(self, g, h, e):
        h = self.embedding_h(h.float())
        if not self.edge_feat:
            e = torch.ones_like(e).to(self.device)
        e = self.embedding_e(e.float())
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        
        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            e = self.MLP_layer(e)
            return {'e': e}
        g.apply_edges(_edge_feat)

        # print("heat-map shape:", y_pred_edges.shape)
        # print("e's shape:", e.shape)
        # print("ndata['h'].shape:", g.ndata['h'].shape)
        # print(g.number_of_nodes())
        # print(y_pred_edges)
            #  
        y_pred_edges = g.edata['e']
        bs_nodes = beamsearch_tour_nodes_shortest(y_pred_edges=y_pred_edges, x_edges_values=g.edata['feat'], beam_size=1280, batch_size=1, num_nodes=g.num_nodes(), dtypeFloat=torch.cuda.FloatTensor, dtypeLong=torch.cuda.LongTensor, probs_type='logits')
        pred_tour_len = mean_tour_len_nodes(g.edata['feat'], bs_nodes)
        # pred_tour_len = self.compute_tour_len(g.edata['e'], g.edata['feat'])
        # print("predicted tour length:", pred_tour_len) 

        return g.edata['e'], pred_tour_len
    
    # def loss(self, pred, label):
    #     criterion = nn.CrossEntropyLoss(weight=None)
    #     loss = criterion(pred, label)

    #     return loss
     
    def loss(self, pred, label, pred_tour, label_tour):
        alpha = 0.9 # fine tuned. Don't ask why.
        criterion1 = nn.CrossEntropyLoss(weight=None)
        criterion2 = nn.L1Loss()
        loss_class = criterion1(pred, label)
        loss_tour = criterion2(pred_tour, label_tour)
        loss = alpha * loss_class + (1 - alpha) * loss_tour  
        return loss
    



        
        
        


#     return 


def mean_tour_len_edges(x_edges_values, y_pred_edges):
    """
    Computes mean tour length for given batch prediction as edge adjacency matrices (for PyTorch tensors).

    Args:
        x_edges_values: Edge values (distance) matrix (batch_size, num_nodes, num_nodes)
        y_pred_edges: Edge predictions (batch_size, num_nodes, num_nodes, voc_edges)

    Returns:
        mean_tour_len: Mean tour length over batch
    """
    y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
    y = y.argmax(dim=3)  # B x V x V
    # Divide by 2 because edges_values is symmetric
    tour_lens = (y.float() * x_edges_values.float()).sum(dim=1).sum(dim=1) / 2
    mean_tour_len = tour_lens.sum().to(dtype=torch.float).item() / tour_lens.numel()
    return mean_tour_len




def beamsearch_tour_nodes_shortest(y_pred_edges, x_edges_values, beam_size, batch_size, num_nodes,
                                   dtypeFloat, dtypeLong, probs_type='raw', random_start=False):
    """
    Performs beamsearch procedure on edge prediction matrices and returns possible TSP tours.
    Final predicted tour is the one with the shortest tour length.
    (Standard beamsearch returns the one with the highest probability and does not take length into account.)
    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
        beam_size: Beam size
        batch_size: Batch size
        num_nodes: Number of nodes in TSP tours
        dtypeFloat: Float data type (for GPU/CPU compatibility)
        dtypeLong: Long data type (for GPU/CPU compatibility)
        probs_type: Type of probability values being handled by beamsearch (either 'raw'/'logits'/'argmax'(TODO))
        random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch
    Returns:
        shortest_tours: TSP tours in terms of node ordering (batch_size, num_nodes)
    """
    if probs_type == 'raw':
        # Compute softmax over edge prediction matrix
        y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        # Consider the second dimension only
        y = y[:, :, :, 1]  # B x V x V
    elif probs_type == 'logits':
        # Compute logits over edge prediction matrix
        y = F.log_softmax(y_pred_edges, dim=1)  # B x V x V x voc_edges # edited from dim=3 to dim=2
        # Consider the second dimension only
        y = y[:, 1]  # B x V x V
        y[y == 0] = -1e-20  # Set 0s (i.e. log(1)s) to very small negative number
    # Perform beamsearch
    beamsearch = Beamsearch(beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type, random_start)
    trans_probs = y.gather(0, beamsearch.get_current_state())
    for step in range(num_nodes - 1):
        beamsearch.advance(trans_probs)
        trans_probs = y.gather(1, beamsearch.get_current_state())
    # Initially assign shortest_tours as most probable tours i.e. standard beamsearch
    ends = torch.zeros(batch_size, 1).type(dtypeLong)
    shortest_tours = beamsearch.get_hypothesis(ends)
    # Compute current tour lengths
    shortest_lens = [1e6] * len(shortest_tours)
    for idx in range(len(shortest_tours)):
        shortest_lens[idx] = tour_nodes_to_tour_len(shortest_tours[idx].cpu().numpy(),
                                                    x_edges_values[idx].cpu().numpy())
    # Iterate over all positions in beam (except position 0 --> highest probability)
    for pos in range(1, beam_size):
        ends = pos * torch.ones(batch_size, 1).type(dtypeLong)  # New positions
        hyp_tours = beamsearch.get_hypothesis(ends)
        for idx in range(len(hyp_tours)):
            hyp_nodes = hyp_tours[idx].cpu().numpy()
            hyp_len = tour_nodes_to_tour_len(hyp_nodes, x_edges_values[idx].cpu().numpy())
            # Replace tour in shortest_tours if new length is shorter than current best
            if hyp_len < shortest_lens[idx] and is_valid_tour(hyp_nodes, num_nodes):
                shortest_tours[idx] = hyp_tours[idx]
                shortest_lens[idx] = hyp_len
    return shortest_tours

def tour_nodes_to_tour_len(nodes, W_values):
    """Helper function to calculate tour length from ordered list of tour nodes.
    """
    tour_len = 0
    for idx in range(len(nodes) - 1):
        i = nodes[idx]
        j = nodes[idx + 1]
        tour_len += W_values[i][j]
    # Add final connection of tour in edge target
    tour_len += W_values[j][nodes[0]]
    return 

def is_valid_tour(nodes, num_nodes):
    """Sanity check: tour visits all nodes given.
    """
    return sorted(nodes) == [i for i in range(num_nodes)]


class Beamsearch(object):
    """Class for managing internals of beamsearch procedure.
    References:
        General: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam.py
        For TSP: https://github.com/alexnowakvila/QAP_pt/blob/master/src/tsp/beam_search.py
    """

    def __init__(self, beam_size, batch_size, num_nodes,
                 dtypeFloat=torch.FloatTensor, dtypeLong=torch.LongTensor, 
                 probs_type='raw', random_start=False):
        """
        Args:
            beam_size: Beam size
            batch_size: Batch size
            num_nodes: Number of nodes in TSP tours
            dtypeFloat: Float data type (for GPU/CPU compatibility)
            dtypeLong: Long data type (for GPU/CPU compatibility)
            probs_type: Type of probability values being handled by beamsearch (either 'raw'/'logits'/'argmax'(TODO))
            random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch
        """
        # Beamsearch parameters
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.num_nodes = num_nodes
        self.probs_type = probs_type
        # Set data types
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        # Set beamsearch starting nodes
        self.start_nodes = torch.zeros(batch_size, beam_size).type(self.dtypeLong)
        if random_start == True:
            # Random starting nodes
            self.start_nodes = torch.randint(0, num_nodes, (batch_size, beam_size)).type(self.dtypeLong)
        # Mask for constructing valid hypothesis
        self.mask = torch.ones(batch_size, beam_size, num_nodes).type(self.dtypeFloat)
        self.update_mask(self.start_nodes)  # Mask the starting node of the beam search
        # Score for each translation on the beam
        self.scores = torch.zeros(batch_size, beam_size).type(self.dtypeFloat)
        self.all_scores = []
        # Backpointers at each time-step
        self.prev_Ks = []
        # Outputs at each time-step
        self.next_nodes = [self.start_nodes]

    def get_current_state(self):
        """Get the output of the beam at the current timestep.
        """
        current_state = (self.next_nodes[-1].unsqueeze(2)
                         .expand(self.batch_size, self.beam_size, self.num_nodes))
        return current_state

    def get_current_origin(self):
        """Get the backpointers for the current timestep.
        """
        return self.prev_Ks[-1]

    def advance(self, trans_probs):
        """Advances the beam based on transition probabilities.
        Args:
            trans_probs: Probabilities of advancing from the previous step (batch_size, beam_size, num_nodes)
        """
        # Compound the previous scores (summing logits == multiplying probabilities)
        if len(self.prev_Ks) > 0:
            if self.probs_type == 'raw':
                beam_lk = trans_probs * self.scores.unsqueeze(2).expand_as(trans_probs)
            elif self.probs_type == 'logits':
                beam_lk = trans_probs + self.scores.unsqueeze(2).expand_as(trans_probs)
        else:
            beam_lk = trans_probs
            # Only use the starting nodes from the beam
            if self.probs_type == 'raw':
                beam_lk[:, 1:] = torch.zeros(beam_lk[:, 1:].size()).type(self.dtypeFloat)
            elif self.probs_type == 'logits':
                beam_lk[:, 1:] = -1e20 * torch.ones(beam_lk[:, 1:].size()).type(self.dtypeFloat)
        # Multiply by mask
        beam_lk = beam_lk * self.mask
        beam_lk = beam_lk.view(self.batch_size, -1)  # (batch_size, beam_size * num_nodes)
        # Get top k scores and indexes (k = beam_size)
        bestScores, bestScoresId = beam_lk.topk(self.beam_size, 1, True, True)
        # Update scores
        self.scores = bestScores
        # Update backpointers
        prev_k = bestScoresId / self.num_nodes
        self.prev_Ks.append(prev_k)
        # Update outputs
        new_nodes = bestScoresId - prev_k * self.num_nodes
        self.next_nodes.append(new_nodes)
        # Re-index mask
        perm_mask = prev_k.unsqueeze(2).expand_as(self.mask)  # (batch_size, beam_size, num_nodes)
        self.mask = self.mask.gather(1, perm_mask)
        # Mask newly added nodes
        self.update_mask(new_nodes)

    def update_mask(self, new_nodes):
        """Sets new_nodes to zero in mask.
        """
        arr = (torch.arange(0, self.num_nodes).unsqueeze(0).unsqueeze(1)
               .expand_as(self.mask).type(self.dtypeLong))
        new_nodes = new_nodes.unsqueeze(2).expand_as(self.mask)
        update_mask = 1 - torch.eq(arr, new_nodes).type(self.dtypeFloat)
        self.mask = self.mask * update_mask
        if self.probs_type == 'logits':
            # Convert 0s in mask to inf
            self.mask[self.mask == 0] = 1e20

    def sort_best(self):
        """Sort the beam.
        """
        return torch.sort(self.scores, 0, True)

    def get_best(self):
        """Get the score and index of the best hypothesis in the beam.
        """
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    def get_hypothesis(self, k):
        """Walk back to construct the full hypothesis.
        Args:
            k: Position in the beam to construct (usually 0s for most probable hypothesis)
        """
        assert self.num_nodes == len(self.prev_Ks) + 1

        hyp = -1 * torch.ones(self.batch_size, self.num_nodes).type(self.dtypeLong)
        for j in range(len(self.prev_Ks) - 1, -2, -1):
            hyp[:, j + 1] = self.next_nodes[j + 1].gather(1, k).view(1, self.batch_size)
            k = self.prev_Ks[j].gather(1, k)
        return hyp



def mean_tour_len_nodes(x_edges_values, bs_nodes):
    """
    Computes mean tour length for given batch prediction as node ordering after beamsearch (for Pytorch tensors).
    Args:
        x_edges_values: Edge values (distance) matrix (batch_size, num_nodes, num_nodes)
        bs_nodes: Node orderings (batch_size, num_nodes)
    Returns:
        mean_tour_len: Mean tour length over batch
    """
    y = bs_nodes.cpu().numpy()
    W_val = x_edges_values.cpu().numpy()
    running_tour_len = 0
    for batch_idx in range(y.shape[0]):
        for y_idx in range(y[batch_idx].shape[0] - 1):
            i = y[batch_idx][y_idx]
            j = y[batch_idx][y_idx + 1]
            running_tour_len += W_val[batch_idx][i][j]
        running_tour_len += W_val[batch_idx][j][0]  # Add final connection to tour/cycle
    return running_tour_len / y.shape[0]