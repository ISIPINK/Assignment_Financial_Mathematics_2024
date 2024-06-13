# Assignment - Financial Mathematics 2024 - Isidoor

The goal of the assignment is to price an American option using three different approaches:

1. Binomial trees
2. Longstaff-Schwarz
3. Deep BSDEs.

The assignment is both applied and theoretical. Here follow some general instructions:

- The assignment should be presented in a Jupyter Notebook in Python
- The main goal is to have some Python code running without bugs and computing the same price for the three different approaches
- The code should be easy to understand, with numerous comments where appropriate
- The code should be consistent between the three approaches (in terms of naming, etc.)
- It is, however, equally important to provide detailed theoretical explanations about each of the methods - I expect some equations here for explaining/deriving each pricing method.
- Using Github Copilot for coding is allowed
- Once the assignment is completed, we will arrange for an oral presentation
- Deadline to finish the assignment is end of May 2024.

Here are some specific instructions for each method:

1. **Binomial tree**: As you mentioned in your email, this is a standard approach, very well documented. The course is sufficient as a resource for implementing the algorithm. Given the numerous implementations available on the internet, please provide a detailed theoretical explanation about the binomial tree approach for pricing American options. You can use what is explained in the lecture notes.

2. **Longstaff-Schwarz** : This is also a standard methodology. Main reference is the original Longstaff-Schwarz paper. Implementations are also available online.

These two parts should be completed without too many problems, relatively fast. If this is not the case, please reach out to me.

3. **BSDEs** : Using a deep BSDE pricer to price an American option is a more recent method, much more involved mathematically and technically as well - as it is using machine learning algorithms. Regarding the theoretical part, I would like you to show how a BSDE for pricing an American option can be derived. Regarding the practical part, I would like you to see if you can use an existing implementation for a Deep BSDE pricer and use it to show that it gives the same value as the first two approaches. You could also show that this approach scales very well when the dimensionality of the problem increases significantly.

This third part is more challenging, so contrarily to the first two parts, I just want to see how far you can go.

I intentionaly leave you some room regarding the resources you want to use, both for the code and the theoretical part of the assignment. Please reach out to me if you want some help with that.

The grade will be attributed based on the results computed in the notebook, the quality of the code, the quality of the theoretical explanations and the oral presentation.
