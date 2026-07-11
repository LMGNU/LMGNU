class ScalarNode:
    """ Tracks a single numerical value, its derivative, and its computational history """
    
    # Mathematical constant e used for exponential operations
    E = 2.718281828459045

    def __init__(self, val, parents=(), operation=''):
        self.data = val
        self.grad = 0.0
        self._backprop = lambda: None
        self._parents = set(parents)
        self._op = operation 

    def __add__(self, secondary):
        secondary = secondary if isinstance(secondary, ScalarNode) else ScalarNode(secondary)
        result = ScalarNode(self.data + secondary.data, (self, secondary), '+')

        def _step_backward():
            self.grad += result.grad
            secondary.grad += result.grad
        result._backprop = _step_backward

        return result

    def __mul__(self, secondary):
        secondary = secondary if isinstance(secondary, ScalarNode) else ScalarNode(secondary)
        result = ScalarNode(self.data * secondary.data, (self, secondary), '*')

        def _step_backward():
            self.grad += secondary.data * result.grad
            secondary.grad += self.data * result.grad
        result._backprop = _step_backward

        return result

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Power must be an integer or float."
        result = ScalarNode(self.data ** exponent, (self,), f'**{exponent}')

        def _step_backward():
            self.grad += (exponent * (self.data ** (exponent - 1))) * result.grad
        result._backprop = _step_backward

        return result

    def relu(self):
        activation = self.data if self.data > 0 else 0
        result = ScalarNode(activation, (self,), 'ReLU')

        def _step_backward():
            self.grad += (1.0 if result.data > 0 else 0.0) * result.grad
        result._backprop = _step_backward

        return result

    def exp(self):
        clipped_data = max(-500, min(500, self.data))
        result = ScalarNode(ScalarNode.E ** clipped_data, (self,), 'exp')

        def _step_backward():
            self.grad += result.data * result.grad
        result._backprop = _step_backward

        return result

    def sigmoid(self):
        clipped_data = max(-500, min(500, self.data))
        val = 1.0 / (1.0 + (ScalarNode.E ** -clipped_data))
        result = ScalarNode(val, (self,), 'sigmoid')

        def _step_backward():
            self.grad += (result.data * (1.0 - result.data)) * result.grad
        result._backprop = _step_backward

        return result

    def tanh(self):
        clipped_data = max(-500, min(500, self.data))
        ep = ScalarNode.E ** clipped_data
        en = ScalarNode.E ** -clipped_data
        val = (ep - en) / (ep + en)
        result = ScalarNode(val, (self,), 'tanh')

        def _step_backward():
            self.grad += (1.0 - result.data ** 2) * result.grad
        result._backprop = _step_backward

        return result

    def backward(self):
        execution_order = []
        seen_nodes = set()
        
        def trace_graph(node):
            if node not in seen_nodes:
                seen_nodes.add(node)
                for parent in node._parents:
                    trace_graph(parent)
                execution_order.append(node)
                
        trace_graph(self)

        self.grad = 1.0
        for node in reversed(execution_order):
            node._backprop()

    def __neg__(self): return self * -1
    def __radd__(self, secondary): return self + secondary
    def __sub__(self, secondary): return self + (-secondary)
    def __rsub__(self, secondary): return secondary + (-self)
    def __rmul__(self, secondary): return self * secondary
    def __truediv__(self, secondary): return self * (secondary ** -1)
    def __rtruediv__(self, secondary): return secondary * (self ** -1)

    def __repr__(self):
        return f"ScalarNode(data={self.data}, grad={self.grad})"


class Tensor:
    """ A basic 2D Tensor wrapping a matrix of ScalarNode objects """

    def __init__(self, data):
        # Initialize matrix converting raw numbers to ScalarNode instances if necessary
        self.data = [
            [x if isinstance(x, ScalarNode) else ScalarNode(x) for x in row]
            for row in data
        ]
        self.shape = (len(self.data), len(self.data[0]) if self.data else 0)

    def __add__(self, other):
        assert self.shape == other.shape, f"Shape mismatch for addition: {self.shape} vs {other.shape}"
        
        res_data = []
        for i in range(self.shape[0]):
            row = []
            for j in range(self.shape[1]):
                row.append(self.data[i][j] + other.data[i][j])
            res_data.append(row)
            
        return Tensor(res_data)

    def matmul(self, other):
        """ Classic Matrix Multiplication: (M x N) x (N x P) -> (M x P) """
        assert self.shape[1] == other.shape[0], f"Cannot multiply matrices of shapes {self.shape} and {other.shape}"
        
        M, N = self.shape
        _, P = other.shape
        
        res_data = []
        for i in range(M):
            row = []
            for j in range(P):
                # Calculate the dot product using ScalarNode operations
                dot_product = ScalarNode(0.0)
                for k in range(N):
                    dot_product += self.data[i][k] * other.data[k][j]
                row.append(dot_product)
            res_data.append(row)
            
        return Tensor(res_data)

    def relu(self):
        """ Element-wise ReLU activation """
        return Tensor([[x.relu() for x in row] for row in self.data])

    def backward(self):
        """ Triggers backward pass on all scalar nodes within the tensor """
        for row in self.data:
            for x in row:
                x.backward()

    def get_data(self):
        """ Helper to view raw forward values easily """
        return [[x.data for x in row] for row in self.data]

    def get_grads(self):
        """ Helper to view raw calculated gradients easily """
        return [[x.grad for x in row] for row in self.data]

    def __repr__(self):
        lines = [str([f"d:{x.data}, g:{x.grad}" for x in row]) for row in self.data]
        return "Tensor([\n  " + ",\n  ".join(lines) + "\n])"