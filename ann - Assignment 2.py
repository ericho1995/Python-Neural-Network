"""
Author: Tai Eric Ho (32118279)

Inspired from the human brain, artificial neural networks (ANNs) are a
type of computer vision model to classify images into certain categories.
In particular, in this assignment we will consider ANNs for the taks of
recognising handwritten digits (0 to 9) from black-and-white images with a
resolution of 28x28 pixels. In Part 1 of this assignment you will create
functions that compute an ANN output for a given input, and in Part 2 you
will write functions to "attack" an ANN. That is, to try to generate inputs
that will fool the network to make a wrong classification.

Part 1 is due at the end of Week 6 and Part 2 is due at the end of week 11.

For more information see the function documentation below and the
assignment sheet.
"""


# Part 1 (due Week 6)
def linear(x, w, b):  # 1 Mark
    """
    Input: A list of inputs (x), a list of weights (w) and a bias (b).
    Output: A single number corresponding to the value of f(x) in Equation 1.

    >>> x = [1.0, 3.5]
    >>> w = [3.8, 1.5]
    >>> b = -1.7
    >>> linear(x, w, b)
    7.35

    This is a processing problem. We are to find the output of the function given the following lists and single integer.
    The function has to provide only 1 singular output using the lists values but also not mutating the given lists.

    In my implementation, I chose to iterate over the 'x' list using a for-loop and to have to seperate variables
    beginning at zero.
    To assist in my iteration, I organised a variable y = 0 to add 1 (y+=1) for each iteration as this allows me
    to jump into the value within the respective list we want to go through. Within the loop we had a variable
    'linear_answer' which is the result from the multiplication of our lists first values then loops in second ...
    until the list is complete then we add our single given integer b.


    """
    y = 0
    linear_answer = 0
    for i in x:
        linear_answer = linear_answer + (i * w[y])
        y += 1
    return linear_answer+b


def linear_layer(x, w, b):  # 1 Mark
    """
    Input: A list of inputs (x), a table of weights (w) and a list of
           biases (b).
    Output: A list of numbers corresponding to the values of f(x) in
            Equation 2.

    >>> x = [1.0, 3.5]
    >>> w = [[3.8, 1.5], [-1.2, 1.1]]
    >>> b = [-1.7, 2.5]
    >>> linear_layer(x, w, b)
    [7.35, 5.15]

    This is a problem where we are to use a given 'list of inputs (x)' multiply it with the respective
    given 'Table of Weights (w)' list elements, add them together and finally add the 'bias (b)' in accordance with
    how many times we've been through the loop.

    In my implement, I used the function we previously made called 'linear()'. This function will be put into a for loop
    and we will use 'i' which will be each list inside the 'w' list. We want the values of list x to stay the same
    but w and b will go to the next item in the list after each time we go through the loop.
    We add the values of each output into our empty list 'z' and then return our completed 'z' list.

    """
    z = []
    y = 0
    for i in w:
        z.append(linear(x,i,b[y]))
        y += 1
    return z


def inner_layer(x, w, b):  # 1 Mark
    """
    Input: A list of inputs (x), a table of weights (w) and a
           list of biases (b).
    Output: A list of numbers corresponding to the values of f(x) in
            Equation 4.

    This problem is telling us to return the maximum number between two numbers after it has applied the previous
    functions. These two numbers is 'the value after applying linear_layer()' and 0. Once we've established the greater
    number we add it to our new list.

    Our Linear() function will give us value given a list of input,weight and singular bias.
    In my implementation, to make code more readable a for longevity. I chose to create a variable within my for-loop
    called 'linear_result' which is our output after applying linear(), in future we can reuse linear_result instead of
    retyping of the entire linear() with the additional variables.
    The code will loop and return a float of 0 if the number outputed from linear_result is less than 0.
    This 0 we will added to our new list.


    >>> x = [1, 0]
    >>> w = [[2.1, -3.1], [-0.7, 4.1]]
    >>> b = [-1.1, 4.2]
    >>> inner_layer(x, w, b)
    [1.0, 3.5]
    >>> x = [0, 1]
    >>> inner_layer(x, w, b)
    [0.0, 8.3]
    """
    y = 0
    newlst = []
    for i in w:
        linear_result = linear(x,i,b[y])
        if (linear_result < 0):
            newlst.append(float(0))
        else:
            newlst.append(linear_result)
        y += 1
    return newlst


def inference(x, w, b):  # 2 Marks
    """
    Input: A list of inputs (x), a list of tables of weights (w) and a table
           of biases (b).
    Output: A list of numbers corresponding to output of the ANN.

    Now we are putting together our ANN. The idea of this function is to apply the previous functions created and
    output our list that we will use for a prediction later on.
    Example how we intend the code to work:
    input -> inner_layer until...-> we reach the last bias then we apply -> linear_layer -> return OUTPUT
    We only want to apply the linear layer when we are on our file bias which will be our Outer layer.

    Problems. We need to replace input with the output after it has applied the inner_layer and how do we know
    when we've reach the last bias.

    We know that inner_layer() will be applied up until the last bias, that is this function. Inner_layer is outputting
    the greater number between it and 0 after it has applied the linear() function. We are looping our function by
    increments of 1, I applied an if-statement that when I reaches the last list inside the bias (b[-1]) then we apply
    our linear_layer (output layer function). I found this saves space and is more convient than b[len(b)-1]
    Up until this  point I am looping the output of inner_layer to replace my X input as after inner_layer has been
    applied we are at that specified layer.


    >>> x = [1, 0]
    >>> w = [[[2.1, -3.1], [-0.7, 4.1]], [[3.8, 1.5], [-1.2, 1.1]]]
    >>> b = [[-1.1, 4.2], [-1.7, 2.5]]
    >>> inference(x, w, b)
    [7.35, 5.15]

    """
    z = []
    for i in range(len(w)):
        if (b[i] == b[-1]):
            x = linear_layer(x,w[i],b[i])
            z = x
        else:
            x = inner_layer(x,w[i],b[i])
            z = x
    return z



def read_weights(file_name):  # 1 Mark
    """
    Input: A string (file_name) that corresponds to the name of the file
           that contains the weights of the ANN.
    Output: A list of tables of numbers corresponding to the weights of
            the ANN.

    We are to read a txt file, grab data from the text and use it to give back the data in list format.
    Our problems are: how will we the data from the file, what do we need to do with the data, the
    data within the file could vary widely ($,#,1,A etc), assuming that we get integers then they will
    still be read as strings and finally we need to somehow differentiate the data into seperate lists.

    Since this is our program we can assume that the file given will only have numbers and hashtags (#). We
    also know that we will be using the data content as it is read from the file.

    Example:  1,2,3
              1,2,3      will be read as 1,2,3 then 1,2,3 not 3,2,1 or 1,1,2,2,.....
                        We don't need to configure the text file.

     My first implementation was to read in the file_name which will be the name of the text file, we will
     then take in all the data (content) and use split() to seperate the strings from one another.
     We will then place a for loops to iterate the: another split whenever we are not in the hashtag line.
     But as the number in content are still strings, we need to change them floats, which is what I applied in the
     nested for loop.
     Upon completion of the nested loop I implemented a method called "append" which will add output into a new list.
     Up to here I had issues of seperating 'i' number of lists to create within a nested nested list. As the variable
     WeightsLst that I use to add the new 'i' list would have an addiction list example: [[],[1,2,3],[1,2,3]]], due to
     the second hashstag if statement. I worked around this by creating an empty list called lst, and then using
     the method to remove (remove()) to change the list from [['',[1,2,3],[1,2,3]]] to our expected output.

    >>> w_example = read_weights('example_weights.txt')
    >>> w_example
    [[[2.1, -3.1], [-0.7, 4.1]], [[3.8, 1.5], [-1.2, 1.1]]]
    >>> w = read_weights('weights.txt')
    >>> len(w)
    3
    >>> len(w[2])
    10
    >>> len(w[2][0])
    16



    """
    x = open(file_name)
    content = x.read()
    content = content.split()
    lst = ''
    WeightsLst = []

    for i in content:
        if i != '#':
            i = i.split(',')
            for j in range(len(i)):
                i[j] = float(i[j])
            lst.append(i)
        if i == '#':
            WeightsLst.append(lst)
            lst = []
    WeightsLst.remove('')
    WeightsLst.append(lst)
    return WeightsLst



def read_biases(file_name):  # 1 Mark
    """
    Input: A string (file_name), that corresponds to the name of the file
           that contains the biases of the ANN.
    Output: A table of numbers corresponding to the biases of the ANN.

    Similar to read_weights function but we can skip a step by doing one less nested list. Our problem is to
    read in the biases text file, change the values to floats and then save them in a new list.

    Therefore very similar to our implementation in read_weights, we will apply a for-loop that upon each iteration
    if what is read is not a hashtag (#) then we'll want to (split()) the data to seperate the strings and then
    apply a nested loop to change them all to floats which will inturn allow us to input into our list by applying
    append().

    Further explanation for the reaosn for the seperation. Each i is a new list of biases, we wouldn't want them all
    in 1 list as we need the input to go through multiple layers and therefore biases.


    >>> b_example = read_biases('example_biases.txt')
    >>> b_example
    [[-1.1, 4.2], [-1.7, 2.5]]
    >>> b = read_biases('biases.txt')
    >>> len(b)
    3
    >>> len(b[0])
    16

    """
    x = open(file_name)
    content = x.read()
    content = content.split()
    lst = []
    for i in content:
        if i != '#':
            i = i.split(',')
            for j in range(len(i)):
                i[j] = float(i[j])
            lst.append(i)
    return lst


def read_image(file_name):  # 1 Mark
    """
    Input: A string (file_name), that corresponds to the name of the file
           that contains the image.
    Output: A list of numbers corresponding to input of the ANN.

    The "image" file is a list of 0's and 1's which will be our input file (X). Our goal is to read the file and return
    a list of all the 0's and 1.

    Problem: all the values inside are strings. A split() function will seperate our inputs into singular strings.

    We are assuming that the input can be any amount of numbers, a while loop would work just as well in this
    scenario but I implemented a nested for-loop which will iterate through every single number in the file, change
    it to a float and add them onto our lst.

    >>> x = read_image('image.txt')
    >>> len(x)
    784
    """

    x = open(file_name)
    content = x.read()
    content = content.split()
    lst = []
    for i in content:
        for j in i:
            j = int(j)
            lst.append(j)
    return lst


def argmax(x):  # 1 Mark
    """
    Input: A list of numbers (i.e., x) that can represent the scores
           computed by the ANN.
    Output: A number representing the index of an element with the maximum
            value, the function should return the minimum index.

    After our inference function we will be outputting a list of numbers and we need to find the highest value
    within the list and output it's position in the list. The issue that may arise is that we may see two of the
    same numbers being the largest of the list. But we only need to return the first maximum that first
    appears in our of the list.

    I implemented a for-loop as it will run the code though in order. I don't need to worry about whether the 9th
    value being the same number as the 2nd as I've apply an if-statement within the loop that if 'i' matched
    the MAX value in the list then return our answer.

    >>> x = [1.3, -1.52, 3.9, 0.1, 3.9]
    >>> argmax(x)
    2
    """

    z = 0
    for i in x:
        if i == max(x):
            return z
        z += 1


def predict_number(image_file_name, weights_file_name, biases_file_name):  # 1 Mark
    """
    Input: A string (i.e., image_file_name) that corresponds to the image
           file name, a string (i.e., weights_file_name) that corresponds
           to the weights file name and a string (i.e., biases_file_name)
           that corresponds to the biases file name.
    Output: The number predicted in the image by the ANN.

    >>> i = predict_number('image.txt', 'weights.txt', 'biases.txt')
    >>> print('The image is number ' + str(i))
    The image is number 4

    This is our call main function that calls all our previously written functions. That reads in the files and
    now we are outputting our prediction. This could will only output given the text files and functions are correct.

    To save space, instead of organising a variable for answer. I implemented a return of argmax() which is the max
    value of the list that comes up first in order of the inference outputs.
    """

    x = read_image(image_file_name)
    b = read_biases(biases_file_name)
    w = read_weights(weights_file_name)

    inference_result = inference(x,w,b)
    return argmax(inference_result)


#Assignment Part 2.

def flip_pixel(x):
    """
    >>> x = 1
    >>> flip_pixel(x)
    0
    >>> x = 0
    >>> flip_pixel(x)
    1

    Firstly, we are assuming that the only x input will be 0 or 1. No other input for x.
    We are to change 1 to 0 and 0 to 1. To make the code even short under our assumption of input 0 or 1, I returned
    the absolute value of x minus 1.
    """
    return abs(x - 1)

def modified_list(i,x):
    """
    >>> x = [1, 0, 1, 1, 0, 0, 0]
    >>> i = 2
    >>> modified_list(i,x)
    [1, 0, 0, 1, 0, 0, 0]
    >>> x = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    >>> i = 5
    >>> modified_list(i,x)
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]

    Current problem is that we want to flip the value of the number in the array x in the position of i.

    As we are only doing 1 single number change, we do not need to do any iterations. Therefore using out previous
    function flip_pixel and flipping the position of 'i' is suffice.

    """
    x[i] = flip_pixel(x[i])
    return x

def compute_difference(x1,x2):
    """
    >>> x1 = [1, 0, 1, 1, 0, 0, 0]
    >>> x2 = [1, 1, 1, 0, 0, 0, 1]
    >>> compute_difference(x1,x2)
    3
    >>> x1 = read_image('another_image.txt')
    >>> x2 = read_image('image.txt')
    >>> compute_difference(x1,x2)
    119
    >>> x1 = read_image('image.txt')
    >>> x2 = read_image('image.txt')
    >>> x2 = modified_list(238,x2)
    >>> x2 = modified_list(210,x2)
    >>> quality = compute_difference(x1,x2)
    >>> print('The quality of the adversarial attack is ' + str(quality))
    The quality of the adversarial attack is 2

    Again the assumption is that the given inputs are a list of 0's and 1's.
    We are trying to find the differences between the two given arrays. If the result of x[i] - y[i] = 0 that would
    mean that there was no difference in the position i. If the value is not 0 then we can conclude there's a difference
    and we make a note of it in.

    I implemented a for loop to go through the entire list of length of x1, using x2 also works as they should have the
    same length. In the for loop I applied a append into my empty list that would input all absolute values of
    (x1[i] - x2[i]). To keep the code short in my return statement I summed the appended empty list.

    """
    emptylst = []
    for i in range(len(x1)):
        emptylst.append((abs(x1[i] - x2[i])))
    return sum(emptylst)



def select_pixel(x, w, b):
    """
    >>> x = read_image('image.txt')
    >>> w = read_weights('weights.txt')
    >>> b = read_biases('biases.txt')
    >>> pixel = select_pixel(x, w, b)
    >>> pixel
    238
    >>> x = modified_list(238,x)
    >>> pixel = select_pixel(x, w, b)
    >>> pixel
    210

    The current situation is that we want to find the pixel in our image text file that if changed causes a change in
    our inference prediction list. This change could be an increase in the second most highest predicted output, a
    decrease in the original predicted output or the two most likely predicted numbers get closer.If changing a pixel
    does not satisfy one of the three above objectives then we will skip it.

    Example:
    Original inference = [-2.166686, -5.867823, -1.673018, -4.412667, 5.710625, -6.022383, -2.174282, 0.378930,
    -2.267785, 3.912823]
    Changing 238 pixel
    New inference = [-2.087871, -5.475471, -1.621999, -4.213924, 4.903939, -6.095823, -2.653019, 0.697776,
    -2.230024, 4.242473]
    In this example the original inference has a best prediction of 4 at 5.710625 and second is 9 at 3.912823
    After the change:
    The new inference has a best prediction of 4 at 4.903939 and second is 9 at 4.242473
    The original inference highest prediction output has decreased and second highest prediction has increased.

    We will continue to return the pixel that has the greatest amount of change in accordance to our objective.
    Our issue is that we need to know when to stop. As we reach a point where the greatest prediction number could go
    4 -> 9 -> 4 -> 9 -> 4.... never ending. Our goal is to stop when the original prediction has changed, in this case
    when the new inference 9 has a better prediction than 4.

    Example:
    [-2.102353, -5.111125, -1.799243, -4.036013, 4.205424, -5.937777, -2.942028, 0.794603, -2.199816, 4.538266].
    9 is now more likely than 4.


    In my implementation, I used a brute force method that checks every pixel one at a time inside a for-loop. Inside
    the for-loop I would produce two inferences (old and new) which I then compared against one another to determine
    whether it meets our original objective.
    In order to keep the code more readable I implemented my own helper functions which were: difference(lst),
    compare(old,new) and greatestvalues(lst).

        TwoLargest(lst) job is to output the two greatest inference value and their positions as a list.
        example: [-2.102353, -5.111125, -1.799243, -4.036013, 4.205424, -5.937777, -2.942028, 0.794603, -2.199816, 4.538266].
        Would output: [[4.538266, 9],[4.205424, 4]]

        Compare(old,new) job is test whether their has been a positive or negative change when comparing the
        TwoLargest(old) and TwoLargest(new) which are the two greatest inference values. If the changes are negative
        which would mean the difference between the two inferences (old and new) has widened due to the pixel change
        which is not what we want. If the change is not negative impact then we output the sum of the positive value.

    In order for us to know which pixel provides the most positive change, all values that return a positive value from
    compare(old,new) will be added to 'impactpixels' list. In the format of [(pixel position / i), compare output]

        greatestvalues(lst) job is to isolate the 'compare output' from it's respective pixel position and return the
        greatest "compare output" from the entire list.

    From here I implemented a final for-loop that looks for the pixel position that matches our greatestvalue output
    and outputs that pixel. If the impactpixel list is 0 then we conclude all pixels from that point give a negative
    change and return -1. We would also return -1 if we reached our goal where the original prediction has changed.

    """
    a = read_image('image.txt')
    c = read_weights('weights.txt')
    d = read_biases('biases.txt')


    #returns the 2 highest predicted numbers
    def TwoLargest(lst):
        order = 0
        order2 = 0
        largestnum = max(lst)
        for i in range(len(lst)):
            if lst[i] == largestnum:
                order = i
                lst[i] = 0
        secondplace = max(lst)
        for i in range(len(lst)):
            if lst[i] == secondplace:
                order2 = i
                lst[i] = 0
        return [[largestnum, order], [secondplace, order2]]

    #this function here is testing whether the highest original predicted is different to the new highest predicted
    #number (testing whether we have reached our goal)
    originalimage = TwoLargest(inference(a, c, d))
    aftermodify = TwoLargest(inference(x, w, b))
    if originalimage[0][1] != aftermodify[0][1]:
        return -1

    #the sum of the difference of the two largest predicted numbers
    def compare(old,new):
        oldbig = TwoLargest(old)
        newbig = TwoLargest(new)
        sum = 0
        testold = oldbig[0][0] - oldbig[1][0] #difference between original two highest predicted inference
        testnew = newbig[0][0] - newbig[1][0] #difference between new two highest predicted inference
        if oldbig[0][1] != newbig[0][1]:
            newbig[0],newbig[1] = newbig[1],newbig[0]
        for i in range(len(oldbig)):
            sum = sum + abs(newbig[i][0] - oldbig[i][0])
        if testnew > testold: #testing for negative change (testing whether gap has widened or not)
            return 0
        return sum

    #this function seperates the i from the number and outputs the largest number.
    def greatestvalue(lst):
        numbers = []
        if len(lst) < 1:
            return 0
        for i in lst:
            numbers.append(i[1])
        return max(numbers)

    #main function (refer to documentation)
    imagelength = len(x)
    impactpixels = []
    for i in range(imagelength):
        old = inference(x,w,b)
        x[i] = flip_pixel(x[i])
        new = inference(x,w,b)
        changevalue = compare(old,new)
        if (changevalue < 0):
            x[i] = flip_pixel(x[i])
        else:
            impactpixels.append([i,changevalue])
            x[i] = flip_pixel(x[i])
    a = greatestvalue(impactpixels)
    if len(impactpixels) < 1:
        return -1
    else:
        for i in impactpixels:
            if a in i:
                return i[0]


def write_image(x, file_name):
    """
    >>> x = read_image('image.txt')
    >>> x = modified_list(238,x)
    >>> x = modified_list(210,x)
    >>> write_image(x,'new_image.txt')

    This is a writing problem, we are to write our modified pixels into a new text file. We are going to need to loop
    the lines are ensure that we press enter after 28 characters.

    In my implementation, I used a for-loop (we know when the loop ends) that would write into the text file one
    character at a time but will insert a "\n" which is a blank line or enter key if the for-loop integer is divisible
    by 28.

    """
    f = open(file_name, "w+")
    for i in range(len(x)):
        if i>0 and i%28 == 0:
            f.write("\n")
        f.write(str(x[i]))
    f.close()

#This function is used to test my code on whether it will print my new expected output and tell me how many pixels
#changed
def tester(x1,x2):
    if x2[0] == -1:
        print('Image was pixel perfect already')
    else:
        write_image(x2,'new_image2.txt')
        print(predict_number('new_image2.txt','weights.txt', 'biases.txt'))
        q = compute_difference(x1,x2)
        print('An adversarial image is found! Total of ' + str(q) + ' pixels were flipped.')

def adversarial_image(image_file_name,weights_file_name,biases_file_name):
    """
    >>> x1 = read_image('image.txt')
    >>> x2 = adversarial_image('image.txt','weights.txt','biases.txt')
    >>> tester(x1,x2)
    9
    An adversarial image is found! Total of 2 pixels were flipped.
    >>> x1 = read_image('another_image.txt')
    >>> x2 = adversarial_image('another_image.txt','weights.txt','biases.txt')
    >>> tester(x1,x2)
    Image was pixel perfect already

    This is combining everything together. If there's a pixel that will make a positive changes / an overall impact
    on our ANN then we want to flip that pixel and keep doing so until there's no more.

    In my implementation if the first selection_pixel(x,w,b) returned -1 then we can conclude that no pixels need to be
    flipped. If there was a pixel then applying a while-loop would be best as we are unsure of how many pixels we will
    be flipping before there are no pixels worth flipping. Therefore we'd get a pixel -> modify our image (not in the
    text file YET) -> search for another pixel using our modified image list and select_pixel(modified image list,w,b)
    -> repeat until we receive -1. Our last modified image list will be adversial image.
    """
    x = read_image(image_file_name)
    w = read_weights(weights_file_name)
    b = read_biases(biases_file_name)

    pixel = select_pixel(x,w,b)
    if pixel != -1:
        while pixel != -1:
            x = modified_list(pixel, x)
            pixel = select_pixel(x,w,b)
        return x
    else:
        return [-1]

if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)