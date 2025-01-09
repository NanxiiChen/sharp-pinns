import torch


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers, norm=False):
        super().__init__()
        self.in_layer = torch.nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.out_layer = torch.nn.Linear(hidden_dim, out_dim)
        self.act = torch.nn.Tanh()
        self.norm = norm
        
    def forward(self, x):
        x = self.act(self.in_layer(x))
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        return self.out_layer(x) if self.norm == False else torch.tanh(self.out_layer(x)) / 2 + 1/2
        

class MultiScaleFeatureFusion(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.low_freq_branch = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim//2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim//2, hidden_dim),
            torch.nn.Tanh()
        )
        self.high_freq_branch = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim*2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim*2, hidden_dim),
            torch.nn.Tanh(),
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        low_freq_features = self.low_freq_branch(x)
        high_freq_features = self.high_freq_branch(x)
        
        combined_features = torch.cat((low_freq_features, high_freq_features), dim=1)
        attention_weights = self.attention(combined_features)
        return attention_weights * low_freq_features \
            + (1 - attention_weights) * high_freq_features


        
class MultiscaleAttentionNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers):
        super().__init__()
        self.feature_fusion = MultiScaleFeatureFusion(in_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        for idx in range(layers):
            self.hidden_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.attention_layers.append(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.Sigmoid()
            ))
        
        self.out_layer = torch.nn.Linear(hidden_dim, out_dim)
        self.act = torch.nn.GELU()
        
    def forward(self, x):
        x = self.feature_fusion(x)
        for layer, attention_layer in zip(self.hidden_layers, self.attention_layers):
            identity = x
            x = self.act(layer(x))
            attention_weights = attention_layer(x)
            x = attention_weights * x + identity
            
        return self.out_layer(x)

        
class ResNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers):
        super().__init__()
        self.in_layer = torch.nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.out_layer = torch.nn.Linear(hidden_dim, out_dim)
        self.act = torch.nn.Tanh()

        
    def forward(self, x):
        x = self.act(self.in_layer(x))
        for layer in self.hidden_layers:
            identity = x
            x = self.act(layer(x)) + identity

        return self.out_layer(x)


class ModifiedMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers, norm=False):
        super().__init__()
        self.gate_layer_1 = torch.nn.Linear(in_dim, hidden_dim)
        self.gate_layer_2 = torch.nn.Linear(in_dim, hidden_dim)
        
        self.hidden_layers = torch.nn.ModuleList([
            torch.nn.Linear(in_dim if idx == 0 else hidden_dim, hidden_dim) for idx in range(layers)
        ])

        self.out_layer = torch.nn.Linear(hidden_dim, out_dim)
        self.act = torch.nn.GELU()
        self.norm = norm
        
        # use xavier initialization
        torch.nn.init.xavier_normal_(self.gate_layer_1.weight)
        torch.nn.init.xavier_normal_(self.gate_layer_2.weight)
        for layer in self.hidden_layers:
            torch.nn.init.xavier_normal_(layer.weight)
        torch.nn.init.xavier_normal_(self.out_layer.weight)

    def forward(self, x):
        u = self.act(self.gate_layer_1(x))
        v = self.act(self.gate_layer_2(x))
        for idx, layer in enumerate(self.hidden_layers):
            x = torch.tanh(layer(x))
            x = x * u + (1 - x) * v
        return self.out_layer(x) if self.norm == False else torch.tanh(self.out_layer(x)) / 2 + 1/2
            
            
class UNetPINN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers):
        super().__init__()
        self.encoder = torch.nn.ModuleList([
            torch.nn.Linear(in_dim if idx == 0 else hidden_dim, hidden_dim) for idx in range(layers)
        ])
        self.decoder = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)
        ])
        self.out_layer = torch.nn.Linear(hidden_dim, out_dim)
        self.act = torch.nn.Tanh()
        
    def forward(self, x):
        skip_connections = []
        
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx < len(self.encoder) - 1:
                skip_connections.append(x)
            x = self.act(x)
            
        for idx, layer in enumerate(self.decoder):
            if idx < len(self.decoder) - 1:
                x = layer(x) + skip_connections.pop()
            else:
                x = layer(x)
            x = self.act(x)
            
        return self.out_layer(x)