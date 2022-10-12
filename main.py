from asyncio.windows_events import NULL
from libraries import *
numpy.random.seed(13)
#picture depth is at index 2 in the tuple
#numpy tuple depth is at index 0
class MaxPooling:
    def __init__(self, input, poolingSize=2):
        self.input = input
        self.poolingSize = poolingSize
        self.result = numpy.zeros((int(self.input.shape[0]/poolingSize),
                                   int(self.input.shape[1]/poolingSize),
                                   self.input.shape[2]))

    def runMaxPooling(self):
        size = int(self.input.shape[0]/self.poolingSize)
        for depth in range(0, self.input.shape[2]):
            for row in range(0, size):
              for col in range(0, size):  
                self.result[row,col,depth] = numpy.max(self.input[row*self.poolingSize: row*self.poolingSize + self.poolingSize,
                                                        col*self.poolingSize: col*self.poolingSize + self.poolingSize,
                                                        depth])
        return self.result

class convolution2D:
    def __init__(self, input, numofFilter, filterSize=3, padding=0, stride=1, poolingSize = 2):
        self.input = numpy.pad(input, (padding, padding),'constant')
        self.stride = stride
        self.poolingSize = poolingSize
        self.filter = numpy.random.randn(numofFilter, filterSize, filterSize)
        self.result = numpy.zeros((int((self.input.shape[0] - filterSize)/self.stride) + 1,
                                   int((self.input.shape[1] - filterSize)/self.stride) + 1, 
                                   numofFilter)) #create zero matrix

    def leakRelu(self, value):
        if value < 0:
            return 0.1*value
        else:
            return value


    def Relu(self, value):
        if value < 0:
            return 0
        else:
            return value
    
    def getROI(self):
        inputEdgeSize = self.input.shape[1]
        filterEdgeSize = self.filter.shape[1]
        for row in range(0, int((inputEdgeSize - filterEdgeSize)/self.stride) + 1):
            for col in range(0, int((inputEdgeSize - filterEdgeSize)/self.stride) + 1):
                roi = self.input[row * self.stride: row * self.stride + filterEdgeSize,
                                col*self.stride: col * self.stride + filterEdgeSize] #numpy array can multiply dot product of matrix
                yield row, col, roi

    
    def runConV2D(self):
        for depth in range(self.filter.shape[0]):
            for row, col, roi in self.getROI():
                self.result[row, col, depth] = self.Relu(sum(sum(roi * self.filter[depth])))
        self.result = self.runMaxPooling(self.result, self.result.shape[0], self.result.shape[1], self.result.shape[2])
        return self.result
    
    
    def runMaxPooling(self, input, inputRow, inputCol, inputDepth):
        tempResult = numpy.zeros((int(inputRow/self.poolingSize),
                                   int(inputCol/self.poolingSize),
                                   inputDepth))
        size = int(inputRow/self.poolingSize)
        for depth in range(0, inputDepth):
            for row in range(0, size):
              for col in range(0, size):  
                tempResult[row,col,depth] = numpy.max(input[row*self.poolingSize: row*self.poolingSize + self.poolingSize,
                                                        col*self.poolingSize: col*self.poolingSize + self.poolingSize,
                                                        depth])
        return tempResult

    

def main():
    image  = cv2.imread("healthy_sample.jpg")
    image = cv2.resize(image, (200,200))   
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255
    conv2D_result_img = convolution2D(gray_img, 16).runConV2D()
    #conv2D_result_img_pooling = MaxPooling(conv2D_result_img,2).runMaxPooling()
    pyplot.figure(figsize=(9,9))
    for i in range(0, 16):
        pyplot.subplot(4,4,i+1)
        pyplot.imshow(conv2D_result_img[:,:,i],cmap='gray')
        pyplot.axis('off')
    
    # result_img = gray_img
    # conv2D = convolution2D(result_img)
    # result_img = conv2D.runConV2D()
    # pyplot.imshow(result_img, cmap='gray')
    # for i in range(0,9):
    #     conv2D = convolution2D(result_img, filterSize=5, padding=2, stride=1)
    #     result_img = conv2D.runConV2D()
    #     pyplot.subplot(3,3,i+1)
    #     pyplot.imshow(result_img, cmap='gray')
    pyplot.savefig('conv2d_gray_img_pooling.jpg')
    pyplot.show()


main()
