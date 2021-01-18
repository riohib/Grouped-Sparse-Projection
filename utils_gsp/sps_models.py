#======================================================================================================
#====================================== Sparse Model Class ===========================================
#=====================================================================================================

class SparseResnet:
    def __init__(self, model):
        self.model = model
    
    def get_layers(self):
        params_d = {}
        for name, param in self.model.named_parameters(): 
            params_d[name] = param
        
        layer_list = [x for x in params_d.keys()]
        return layer_list

    def get_layerwise_sps(self):
        """
        This function will output a list of layerwise sparsity of a LeNet-5 CNN model.
        The sparsity measure is the Hoyer Sparsity Measure.
        """
        counter = 0
        weight_d = {}
        sps_d = {}
        for name, param in self.model.named_parameters(): 
            if 'weight' in name:
                if param.dim() > 2:  #Only two different dimension for LeNet-5
                    w_shape = param.shape
                    dim_1 = w_shape[0] * w_shape[1]
                    weight_d[counter] = param.detach().view(dim_1,-1)
                    sps_d[name] = self.sparsity(weight_d[counter]).item()
                else:
                    sps_d[name] = self.sparsity(param.data).item()
                counter += 1
        
        w_name_list = [x for x in sps_d.keys()] 
        return sps_d
    
    def sparsity(matrix):
        ni = matrix.shape[0]

        # Get Indices of all zero vector columns.
        zero_col_ind = (abs(matrix.sum(0) - 0) < 2.22e-16).nonzero().view(-1)  
        spx_c = (np.sqrt(ni) - torch.norm(matrix,1, dim=0) / torch.norm(matrix,2, dim=0)) / (np.sqrt(ni) - 1)
        if len(zero_col_ind) != 0:
            spx_c[zero_col_ind] = 1  # Sparsity = 1 if column already zero vector.
        
        if matrix.dim() > 1:   
            sps_avg =  spx_c.sum() / matrix.shape[1]
        elif matrix.dim() == 1:  # If not a matrix but a column vector!
            sps_avg =  spx_c    
        return sps_avg