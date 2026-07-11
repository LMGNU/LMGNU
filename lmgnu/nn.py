import random


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
        secondary = secondary if isinstance(
            secondary, ScalarNode) else ScalarNode(secondary)
        result = ScalarNode(self.data + secondary.data, (self, secondary), '+')

        def _step_backward():
            self.grad += result.grad
            secondary.grad += result.grad
        result._backprop = _step_backward

        return result

    def __mul__(self, secondary):
        secondary = secondary if isinstance(
            secondary, ScalarNode) else ScalarNode(secondary)
        result = ScalarNode(self.data * secondary.data, (self, secondary), '*')

        def _step_backward():
            self.grad += secondary.data * result.grad
            secondary.grad += self.data * result.grad
        result._backprop = _step_backward

        return result

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)
                          ), "Power must be an integer or float."
        result = ScalarNode(self.data ** exponent, (self,), f'**{exponent}')

        def _step_backward():
            self.grad += (exponent * (self.data **
                          (exponent - 1))) * result.grad
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
        if isinstance(data, list):
            # Check if input is a 1D list, if so convert it to a 2D row matrix [[...]]
            if len(data) > 0 and not isinstance(data[0], list):
                data = [data]
            self.data = [
                [x if isinstance(x, ScalarNode) else ScalarNode(x)
                 for x in row]
                for row in data
            ]
        else:
            raise ValueError(
                "Data must be a nested list representing a matrix.")

        self.shape = (len(self.data), len(self.data[0]) if self.data else 0)

    def __add__(self, other):
        # Element-wise tensor addition or tensor + bias row broadcast
        if self.shape == other.shape:
            res_data = [[self.data[i][j] + other.data[i][j]
                         for j in range(self.shape[1])] for i in range(self.shape[0])]
        elif self.shape[1] == other.shape[1] and other.shape[0] == 1:
            # Broadcast bias row across all rows of self
            res_data = [[self.data[i][j] + other.data[0][j]
                         for j in range(self.shape[1])] for i in range(self.shape[0])]
        else:
            raise ValueError(
                f"Shape mismatch for addition: {self.shape} vs {other.shape}")

        return Tensor(res_data)

    def matmul(self, other):
        """ Classic Matrix Multiplication: (M x N) x (N x P) -> (M x P) """
        assert self.shape[1] == other.shape[
            0], f"Cannot multiply shapes {self.shape} and {other.shape}"

        M, N = self.shape
        _, P = other.shape

        res_data = []
        for i in range(M):
            row = []
            for j in range(P):
                dot_product = ScalarNode(0.0)
                for k in range(N):
                    dot_product += self.data[i][k] * other.data[k][j]
                row.append(dot_product)
            res_data.append(row)

        return Tensor(res_data)

    def relu(self):
        return Tensor([[x.relu() for x in row] for row in self.data])

    def backward(self):
        for row in self.data:
            for x in row:
                x.backward()

    def get_data(self):
        return [[x.data for x in row] for row in self.data]

    def get_grads(self):
        return [[x.grad for x in row] for row in self.data]

    def __repr__(self):
        lines = [str([f"d:{x.data:.4f}, g:{x.grad:.4f}" for x in row])
                 for row in self.data]
        return "Tensor([\n  " + ",\n  ".join(lines) + "\n])"


# --- Network Modules ---

class LearningModule:
    """ Base structural module matching your original setup """

    def reset_gradients(self):
        for parameter in self.get_parameters():
            parameter.grad = 0.0

    def get_parameters(self):
        return []


class LinearUnit(LearningModule):
    """ Replaces Neuron: Operates completely natively using ScalarNode """

    def __init__(self, inputs_count, apply_nonlin=True):
        self.weights = [ScalarNode(random.uniform(-1, 1))
                        for _ in range(inputs_count)]
        self.bias = ScalarNode(0.0)
        self.apply_nonlin = apply_nonlin

    def __call__(self, x):
        # x is a list of scalar objects or numbers
        activation = sum(
            (w_i * x_i for w_i, x_i in zip(self.weights, x)), self.bias)
        return activation.relu() if self.apply_nonlin else activation

    def get_parameters(self):
        return self.weights + [self.bias]

    def __repr__(self):
        return f"{'ReLU' if self.apply_nonlin else 'Linear'}Unit({len(self.weights)})"


class DenseLayer(LearningModule):
    """ Replaces Layer: Composes multiple LinearUnits and presents them as a unified Tensor operation """

    def __init__(self, inputs_count, outputs_count, **kwargs):
        self.units = [LinearUnit(inputs_count, **kwargs)
                      for _ in range(outputs_count)]

    def __call__(self, x):
        # Handles input whether it arrives as a Tensor or raw list/matrix
        if isinstance(x, Tensor):
            # Process sample by sample if multiple batches exist
            out_matrix = []
            for row in x.data:
                out_matrix.append([unit(row) for unit in self.units])
            return Tensor(out_matrix)
        else:
            outputs = [unit(x) for unit in self.units]
            return Tensor(outputs)

    def get_parameters(self):
        return [param for unit in self.units for param in unit.get_parameters()]

    def __repr__(self):
        return f"DenseLayer of [{', '.join(str(u) for u in self.units)}]"


class SequentialNetwork(LearningModule):
    """ Replaces MLP: Chains multiple DenseLayers together """

    def __init__(self, inputs_count, layers_shapes):
        sizes = [inputs_count] + layers_shapes
        self.layers = [
            DenseLayer(sizes[i], sizes[i+1],
                       apply_nonlin=(i != len(layers_shapes) - 1))
            for i in range(len(layers_shapes))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_parameters(self):
        return [param for layer in self.layers for param in layer.get_parameters()]

    def __repr__(self):
        return f"SequentialNetwork of [{', '.join(str(layer) for layer in self.layers)}]"
