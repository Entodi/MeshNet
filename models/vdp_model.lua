require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'

-- number of layers
n_layers = 8

-- input 4th dimension
input = {1, 21, 21, 21, 21, 21, 21, 21}

-- output 4th dimension
output = {21, 21, 21, 21, 21, 21, 21, 3}

-- kernel size for layers from 1 to 8
kZ = {3, 3, 3, 3, 3, 3, 3, 1}
kY = kZ
kX = kZ

-- default convolution step
dZ = 1
dY = dZ
dX = dZ

-- default padding
padZ = {1, 1, 2, 4, 8, 16, 1, 0}
padY = padZ
padX = padZ

-- dilation value for layers from 1 to 8
dilZ = {1, 1, 2, 4, 8, 16, 1, 1}
dilY = dilZ
dilX = dilZ

-- dropout p
p = 0.25

-- building net architecture
local net = nn.Sequential()
for i = 1, n_layers do
  if i ~= n_layers then
    net:add(nn.VolumetricDilatedConvolution(input[i], output[i],
      kZ[i], kY[i], kX[i],
      dZ, dY, dX,
      padZ[i], padY[i], padX[i],
      dilZ[i], dilY[i], dilX[i]))
    net:add(cudnn.ReLU(true))
    net:add(cudnn.VolumetricBatchNormalization(output[i]))
    net:add(nn.VolumetricDropout(p))
  else
    net:add(nn.VolumetricDilatedConvolution(input[i], output[i],
      kZ[i], kY[i], kX[i],
      dZ, dY, dX,
      padZ[i], padY[i], padX[i],
      dilZ[i], dilY[i], dilX[i]))
  end
end

-- enable cuda mode
net = net:cuda()

-- show architecture
print(net)

return net
