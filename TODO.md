Train on wikipedia. Then ask questions that it answers correctly and those it hallucinates. 

Then use a linear probe on the activations to see if you can tell a hallucinated answer from a correct one (and from an incorrect one where it misunderstood).

Multi GPU training!!!

Think I'm going to do a get_vocab_size again

Next Steps:

Be able to use both encoders. The full tokenizer makes it hard to see whether it's working or not
