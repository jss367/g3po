Train on wikipedia. Then ask questions that it answers correctly and those it hallucinates. 

Then use a linear probe on the activations to see if you can tell a hallucinated answer from a correct one (and from an incorrect one where it misunderstood).

Multi GPU training!!!

Think I'm going to do a get_vocab_size again

Next Steps:

FIX THE ISSUE WHERE THE LOSS IS HIGHER AFTER I LOAD IT

Don't reload and re-split the data each step

Be able to use both encoders. The full tokenizer makes it hard to see whether it's working or not

I'm currently doing it by number of iterations, but would be good to switch to epochs
