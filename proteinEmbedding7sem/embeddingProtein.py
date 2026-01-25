class GAT1(nn.Module):
    def __init__(self, gat_args):
        super(GAT1, self).__init__()
        # under_module_arg = {'hidden_dim': 128, 'feature_dim': 34}
        hidden_dim = gat_args["hidden_dim"]
        self.conv1 = GATConv(hidden_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim)

        self.pool1 = SAGPooling(hidden_dim, dropout=0.4)
        self.pool2 = SAGPooling(hidden_dim, dropout=0.4)
        self.pool3 = SAGPooling(hidden_dim, dropout=0.4)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.fc1 = nn.Linear(gat_args["feature_dim"], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.A = nn.Parameter(torch.randn(hidden_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, batch_data):
        x_feature = batch_data.x
        edge_index = batch_data.edge_index
        edge_distance = batch_data.edge_attr
        batch = batch_data.batch
        x = self.fc1(x_feature)
        # end = global_mean_pool(x, batch)

        x_gnn = self.conv1(x, edge_index, edge_attr=edge_distance)
        x_gnn = self.bn1(x_gnn)
        x = x + x_gnn
        x = F.relu(x)

        y = self.pool1(x, edge_index, batch=batch)
        x = y[0]
        edge_index = y[1]
        batch = y[3]

        x = self.fc2(x)
        x_gnn = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x + x_gnn
        x = F.relu(x)

        y = self.pool2(x, edge_index, batch=batch)
        x = y[0]
        edge_index = y[1]
        batch = y[3]

        x = self.fc3(x)
        x_gnn = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x + x_gnn

        y = self.pool3(x, edge_index, batch=batch)
        x = y[0]
        edge_index = y[1]
        batch = y[3]

        # x = F.relu(x)
        return global_mean_pool(x, batch)
        # return x










class GAT1(nn.Module):
    def __init__(self, gat_args):
        super(GAT1, self).__init__()
        # under_module_arg = {'hidden_dim': 128, 'feature_dim': 34}
        hidden_dim = gat_args["hidden_dim"]
        self.conv1 = GATConv(hidden_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim)

        self.pool1 = SAGPooling(hidden_dim)
        self.pool2 = SAGPooling(hidden_dim)
        self.pool3 = SAGPooling(hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.fc1 = nn.Linear(gat_args["feature_dim"], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.A = nn.Parameter(torch.randn(hidden_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, batch_data):
        x_feature = batch_data.x
        edge_index = batch_data.edge_index
        edge_distance = batch_data.edge_attr
        batch = batch_data.batch
        x = self.fc1(x_feature)
        # end = global_mean_pool(x, batch)

        x_gnn = self.conv1(x, edge_index, edge_attr=edge_distance)
        x_gnn = self.bn1(x_gnn)
        x = x + x_gnn
        x = F.relu(x)

        y = self.pool1(x, edge_index, batch=batch)
        x = y[0]
        edge_index = y[1]
        batch = y[3]

        x = self.fc2(x)
        x_gnn = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x + x_gnn
        x = F.relu(x)

        y = self.pool2(x, edge_index, batch=batch)
        x = y[0]
        edge_index = y[1]
        batch = y[3]

        x = self.fc3(x)
        x_gnn = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x + x_gnn

        y = self.pool3(x, edge_index, batch=batch)
        x = y[0]
        edge_index = y[1]
        batch = y[3]

        # x = F.relu(x)
        return global_mean_pool(x, batch)
        # return x