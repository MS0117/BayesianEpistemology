VERBAL_GPT_TEMPLATE="""Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question based on the evidence. 
Give ONLY the guess and probability, no other words or explanation. 
For example
Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>
Probability: <the probability between 0.0 and 1.0 that your guess is correct based on the given evidence , without any extra commentary whatsoever; just the probability!>
###The question: {question}
###The evidence: {evidence}
"""

DIRECT_VERBAL_GPT_TEMPLATE="""Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. 
Give ONLY the guess and probability, no other words or explanation. 
For example
Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>
Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>
###The question: {question}
"""


TOKEN_GPT_TEMPLATE="""
Provide your best guess for the following question based on the evidence. 
Give ONLY the guess, no other words or explanation. 
For example
Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>
###The question: {question}
###The evidence: {evidence}

"""

DIRECT_TOKEN_GPT_TEMPLATE="""
Provide your best guess for the following question. 
Give ONLY the guess, no other words or explanation. 
For example
Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>
###The question: {question}
"""



SAMPLE_GPT_TEMPLATE="""
Provide your best guess for the following question based on the evidence. 
Give ONLY the guess, no other words or explanation. 
For example
Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>
###The question: {question}
###The evidence: {evidence}
"""
DIRECT_SAMPLE_GPT_TEMPLATE="""
Provide your best guess for the following question. 
Give ONLY the guess, no other words or explanation. 
For example
Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>
###The question: {question}
"""


EXTRACT_ANSWER_FREQUENCY="""
###Candidates:
1. Chris Hoy
2. Steve Redgrave
3. Chris Hoy
4. Chris Hoy
5. Steve Redgrave
6. Sir Chris Hoy
7. Chris Hoy
8. Jason Kenny
9. Steve Redgrave
10. <Sir Chris Hoy>

Extract the most frequency candidate and its frequency . You should handle the synonyms as a same response

###Answer: Chris Hoy
###Frequency: 6


###Candidates:
1. Alliteration
2. tongue twister
3. tongue twister
4. alliteration
5. Tongue twister
6. tongue twister
7. tongue twister
8. alliteration
9. alliteration
10. alliteration

Extract the most frequency candidate and its frequency . You should handle the synonyms as a same response

###Answer: alliteration
###Frequency: 5

###Candidates:
1. 12 penguins
2. 12 penguins
3. 12 penguins
4. 12
5. 12
6. 12 penguins
7. 12
8. 12 
9. 12 
10. 12

Extract the most frequency candidate and its frequency . You should handle the synonyms as a same response

###Answer: 12
###Frequency: 10

###Candidates:
1. intensive
2. Intensive
3. Intensive property
4. intensive property
5. Intensive
6. Intensive
7. intensive
8. intensive
9. Intensive
10. Intensive

Extract the most frequency candidate and its frequency . You should handle the synonyms as a same response

###Answer: Intensive
###Frequency: 10


###Candidates:
1. {source1} 
2. {source2}
3. {source3}
4. {source4}
5. {source5}
6. {source6}
7. {source7}
8. {source8}
9. {source9}
10. {source10}

Extract the most frequency candidate and its frequency . You should handle the synonyms as a same response.


"""


VERBAL_LLAMA_TEMPLATE="""

"""

DIRECT_VERBAL_LLAMA_TEMPLATE="""
"""

VERBAL_PHI_TEMPLATE="""

"""
DIRECT_VERBAL_PHI_TEMPLATE="""
"""

TOKEN_LLAMA_TEMPLATE="""
"""

DIRECT_TOKEN_LLAMA_TEMPLATE="""
"""
TOKEN_PHI_TEMPLATE="""

"""

DIRECT_TOKEN_PHI_TEMPLATE="""
"""

SAMPLE_LLAMA_TEMPLATE="""

"""

DIRECT_SAMPLE_LLAMA_TEMPLATE="""
"""

SAMPLE_PHI_TEMPLATE="""

"""

DIRECT_SAMPLE_PHI_TEMPLATE="""
"""


