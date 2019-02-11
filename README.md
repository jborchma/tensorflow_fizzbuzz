# Tensorflow Fizz Buzz

[Fizz buzz](https://en.wikipedia.org/wiki/Fizz_buzz) is a very old and widely used programming
interview question. It has spawned many awesome solutions and parodies, where the two that
stand out to me are the [enterprise Fizz Buzz](https://github.com/EnterpriseQualityCoding/FizzBuzzEnterpriseEdition) repo as well as Joel Grus' [tensorflow Fizz Buzz](http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/).

While reading Joel's solution I was wondering if it were possible to fully teach the neural network
such that it would be able to reproduce the correct solution perfectly.

## Why should or could this work?

Effectively, the function we want to approximate is based on the modulo function. This function is
a very basic, stepwise continuous function that takes the shape of a seesaw function. Of course,
Fizz Buzz is not just simply a modulo function, but a combination of the linear function `f(x) = x`
and three modulo functions with some logic attached to them. Additionally, the values of the function
are not any numerical values, but for all intents and purposes we could imagine the special value to
just map to three different negative numbers or even some a vector in `R^n` that's not just the
real axis in one dimensions.

My belief that this should be possible was actually based on the [universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem), which states


> [T]he universal approximation theorem states that a feed-forward network with a single hidden layer  containing a finite number of neurons can approximate continuous functions on compact subsets of Rn, under mild assumptions on the activation function.

It was [originally](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.441.7873&rep=rep1&type=pdf) shown for sigmoid activation functions, but has since been shown to be valid for a wide range of
different activation functions.

Of course, the function we want to approximate does not fully fulfill all the conditions of the theorem:
the function is not continuous. Hence, it is not immediately clear that we would be able to
perfectly approximate the Fizz Buzz function. However, its shape is still decently simple that
I have hope to be able to fully teach a neural network the Fizz Buzz rules.

## Improving the network

Joel's approach was already decently successful, however his learned model did not fully succeed in
reproducing the right sequence. There were a few things about the training data and the neural architecture that I changed in order to improve the model:

- the training data set: Neural network shine in problems with a LOT of training data. Hence, I
  increased the training set from the numbers starting at 101 up to 1024 to the range [101,4096]. One could argue that this is still not a lot, but it turned out to be enough for our use case.
- In order to accommodate the increased input range, I had to increase the numbers of binary digits
- I also increased the batch size to 250 instead of 128
- It turned out that we needed an increased number of hidden nodes in order to learn the functions
  pattern (from 100 to 200)

## Results

```console
foo@bar:~$ python fizzbuzz_tf.py
Epoch: 1, train accuracy: 0.5335335335335335, epoch loss: 20.365147829055786
Epoch: 2, train accuracy: 0.5335335335335335, epoch loss: 18.559097170829773
Epoch: 3, train accuracy: 0.5335335335335335, epoch loss: 18.453301548957825
Epoch: 4, train accuracy: 0.5335335335335335, epoch loss: 18.446383833885193
Epoch: 5, train accuracy: 0.5335335335335335, epoch loss: 18.443661332130432
Epoch: 6, train accuracy: 0.5335335335335335, epoch loss: 18.455499053001404
...
Epoch: 2000, train accuracy: 1.0, epoch loss: 0.07465453259646893
['1' '2' 'fizz' '4' 'buzz' 'fizz' '7' '8' 'fizz' 'buzz' '11' 'fizz' '13'
 '14' 'fizzbuzz' '16' '17' 'fizz' '19' 'buzz' 'fizz' '22' '23' 'fizz'
 'buzz' '26' 'fizz' '28' '29' 'fizzbuzz' '31' '32' 'fizz' '34' 'buzz'
 'fizz' '37' '38' 'fizz' 'buzz' '41' 'fizz' '43' '44' 'fizzbuzz' '46' '47'
 'fizz' '49' 'buzz' 'fizz' '52' '53' 'fizz' 'buzz' '56' 'fizz' '58' '59'
 'fizzbuzz' '61' '62' 'fizz' '64' 'buzz' 'fizz' '67' '68' 'fizz' 'buzz'
 '71' 'fizz' '73' '74' 'fizzbuzz' '76' '77' 'fizz' '79' 'buzz' 'fizz' '82'
 '83' 'fizz' 'buzz' '86' 'fizz' '88' '89' 'fizzbuzz' '91' '92' 'fizz' '94'
 'buzz' 'fizz' '97' '98' 'fizz' 'buzz']
Number of correct predictions: 100
Incorrect predictions:  []
```

This looks absolutely great! Now, did we just learn a universal fizz buzz function? Let's try some
higher numbers and check how good our model is. Uncommenting the respective parts in `fizzbuzz_tf.py`
will yield

```console
foo@bar:~$ python fizzbuzz_tf.py
...

Number of correct predictions: 226
Incorrect predictions:  [('fizz', '4098'), ('4099', 'fizz'), ('buzz', '4100'), ('fizz', 'buzz')
...
```

Obviously, our model performs terribly! Only 226 correct predictions out of almost 1000. How could
this be? The reason for this lies in the way we encoded our integers and the training set. Our training
set only included numbers that have a binary representation with up 12 digits (and 4096 with 13).
Hence, none of the other inputs used any of the neurons that would be used for the numbers with
higher number of digits. So those connections were never training and hence our model performs terribly.

How could we solve this? One way to solve this would be to choose a different encoding for our integers
that doesn't suffer such an issues. For example using the normal decimal encoding instead of binary
would make our model range much larger depending on our training set. 
