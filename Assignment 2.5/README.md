# EVA8

## Assignment 2.5 

###  Required Packages

  torch, torchvision, numpy, matplotlib
  

###  Dataset ###

   MNIST from torchvision
  
###  Data Generation ###
   
   ####Preparing Dataset####
   
   1. Generating random number using torch.randint() function
   2. Getting image and label from MNIST data
   3. Calculating label using sum of image label and random number generated
   4. Returning Two inputs and Two labels for Training
   
### Network Architecture ###

1. Flatten image (28 * 28 --> 784) and concatenate with random number (one hot encoded) 

2.   Concatenated input size : 784 + 10 = 794

3.   Pass concatenated data through feed forward NN ( With Relu activation ) (Not used any CNNs)

4.   use output from self.fc2 to get out1 and out2 

5.   self.fc2 -->  self.fc1_out1  --> self.fc2_out1 --> out1

6.   self.fc2 -->  self.fc1_out2  --> out2

7.   Return out1 and out2


```py
# Model Architecture

Mnist_ff(
  (fc1): Linear(in_features=794, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc1_out1): Linear(in_features=64, out_features=32, bias=True)
  (fc2_out1): Linear(in_features=32, out_features=10, bias=True)
  (fc1_out2): Linear(in_features=64, out_features=19, bias=True)
)
```

### Trainig Log:
```
loss at epoch: 0  is  821.527099609375
loss at epoch: 1  is  227.16062927246094
loss at epoch: 2  is  162.1258087158203
loss at epoch: 3  is  143.1830291748047
loss at epoch: 4  is  126.12157440185547
loss at epoch: 5  is  120.95542907714844
loss at epoch: 6  is  114.85391998291016
loss at epoch: 7  is  102.51146697998047
loss at epoch: 8  is  102.35953521728516
loss at epoch: 9  is  99.97547912597656
loss at epoch: 10  is  91.51361083984375
loss at epoch: 11  is  92.4646987915039
loss at epoch: 12  is  90.44698333740234
loss at epoch: 13  is  86.82398986816406
loss at epoch: 14  is  79.69280242919922
loss at epoch: 15  is  78.66291046142578
loss at epoch: 16  is  78.4300308227539
loss at epoch: 17  is  80.95767211914062
loss at epoch: 18  is  78.66265869140625
loss at epoch: 19  is  71.2935562133789
loss at epoch: 20  is  69.60258483886719
loss at epoch: 21  is  72.84759521484375
loss at epoch: 22  is  64.9349365234375
loss at epoch: 23  is  70.33512878417969
loss at epoch: 24  is  62.56410598754883
loss at epoch: 25  is  61.201778411865234
loss at epoch: 26  is  63.401771545410156
loss at epoch: 27  is  61.67580795288086
loss at epoch: 28  is  65.59944152832031
loss at epoch: 29  is  69.1248779296875
loss at epoch: 30  is  51.8309211730957
loss at epoch: 31  is  63.584716796875
loss at epoch: 32  is  59.12314987182617
loss at epoch: 33  is  56.69247817993164
loss at epoch: 34  is  61.10468292236328
loss at epoch: 35  is  61.75617980957031
loss at epoch: 36  is  50.8314323425293
loss at epoch: 37  is  51.03319549560547
loss at epoch: 38  is  61.197181701660156
loss at epoch: 39  is  65.8266372680664
loss at epoch: 40  is  53.390655517578125
loss at epoch: 41  is  55.2442741394043
loss at epoch: 42  is  56.48125076293945
loss at epoch: 43  is  54.51993179321289
loss at epoch: 44  is  52.94220733642578
loss at epoch: 45  is  51.080806732177734
loss at epoch: 46  is  61.020294189453125
loss at epoch: 47  is  57.92759704589844
loss at epoch: 48  is  59.070613861083984
loss at epoch: 49  is  53.256221771240234

```

### Predictions ###

``` 
images correctly predicted are : 5773 
 correctly predicted sums are :   5339
```
   



