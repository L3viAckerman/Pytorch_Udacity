import torch 

t = torch.Tensor(2, 3, 4)
type(t)
print(t)
t.zero_()
print(t)
t.resize_(3, 8)
print(t)

v = torch.Tensor([[2, 3, 4],[1, 2,0]])
w = torch.Tensor([[3, 1, 0], [3, 4, 5]])

#Nhân 
print(v*w)
 
a = torch.Tensor([2, 3, 4])
b = torch.Tensor([1, 4, -1])
print(a.dot(b))
print(a @ b)

a = torch.Tensor(5).random_(10)
print(a)

a = torch.arange(1, 4 + 1)
print(a)
print(torch.arange(10))
print(a.pow(2))     #Square all elements in the tensor 

# Matrix (2D Tensor)

# Create a 2*4  tensor
m = torch.Tensor([[2, 4, 3, 7],[4, 2, 1, 9]])
print(m)

print(m.dim())

print(m.size(), m.size(1), m.size(0), sep=' -- ')

# Returns the total number of elements, hence numel (number of elements)
print(m.numel())

# Indexing row 0, column 2 (0-indexed)
print(m[0, 2])

# Indexing row 0, column 2 (0-indexed)
print(m[0][2])

# Indexing column 1, all rows (return size 2)
print(m[:, 1])

# Indexing column 1, all rows (return size 2*2)
print(m[:, [1]])

# Indexing row 0, all columns (return size 1*4)
print(m[[0], :])

# Indexing row 0, all columns (return size 4)
print(m[0, :])

# Create tensor of numbers from 1 to 5 (excluding 5)
v = torch.arange(1., 4+1)
print(v)

# Scalar product
print(m @ v)

# Calculated by 1 * 2 + 2 *5 + 3 * 3 + 4 * 7
print(m[[0], :] @ v)

# Calculate 
print(m[[1], :] @ v)

# Add a random tensor of size 2*4 to m
m + torch.rand(2, 4)

# Subtract a random tensor of size 2*4 to m
print(m - torch.rand(2, 4))

# Multiply a random tensor of size 2*4 to m
m * torch.rand(2, 4)

# Divide m by a random tensor of size 2*4
print(m / torch.rand(2, 4))

print(m.size())

# Transpose tensor m
print(m.t())


# Constructors

# Create tensor from 3 to 8, with each having a space of 1
torch.arange(3, 8 + 1)


# Create tensor form 5.7 to -2.1 with each having a space of -3
print(torch.arange(5.7, -2.1, -3))

# Return a 1D tensor of steps equally spaced points between start = 3, end =8 and steps = 20
print(torch.linspace(3, 8, 20).view(1, -1))

# Create a tensor filled with 0's
torch.zeros(3, 5)

# Create a tensor filled with 1's
v = torch.ones(3, 2, 5)
print(v)

# Create a tensor with the diagonal filled with 1
print(torch.eye(3))

# Set default plots
# from plot_lib import set_default
from matplotlib import pyplot as plt 
# set_default()

print(plt.hist(torch.randn(1000).numpy(), 100))

# Casting 

# This i basically a 64 bit float tensor
m_double = m.double()
print(m_double)

# Move your tensor to GPU device 0 if there is one (if GPU is in the system)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
m.to(device)
print(m)

# Converts tensor to numpy array
m_np = m.numpy()
print(m_np)

# In-place fill of column 0 and row 0 with value -1
m_np[0, 0] = -1
print(m_np)

print(m)        # Gia tri cua m_np thay doi thi m cung thay doi

# Create a tensor of integers ranging from 0 to 4
import numpy as np 
n_np = np.arange(5)
n = torch.from_numpy(n_np)
print(n, n_np)

# In-place multiplication of all elements by 2 for tensor n
# Because n is essentiall n_np, not a clone, this affects n_np
print(n.mul_(2))
print(n_np)


# More fun

# Concatenate on axis 0, so u get 2 * 4
a = torch.Tensor([[1, 2, 3, 4]])
b = torch.Tensor([[5, 6, 7, 8]])

c = torch.cat((a, b), 0)
print(c)

# Concatenate on axis 1, so u get 1*8
d = torch.cat((a, b), 1)