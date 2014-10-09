def clip_element_wise(arr, minvals, maxvals):
    """
    min vals are 5 element, maxvals is all
    """
    for i in xrange(arr.size):
        arr[i] = arr[i].clip(min=minvals[i],max=maxvals[i])

