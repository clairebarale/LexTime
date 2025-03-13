##################################################
## Examples and Prompts for the Inference task
##################################################


# Examples
example1 = f"""
        context: "Before assigning the plaintiff to work, the defendant hired employees, including the plaintiff. The plaintiff was hired as a trackman."

        query: "Does hiring employees happens before assigning the plaintiff to work?"

        answer: "yes"
        """
    
example2 = f"""
        context: "ms. reed advised ms. salome to issue the written discipline and to speak with rivera (as well as the other roll tunnel operators) regarding the defaced guidelines. accordingly, ms. salome, along with clint kelch (the union steward) approached rivera to present him with the disciplinary warning and to discuss the defacement of the posted guidelines. rivera was on his clamp truck and ms. salome advised that rivera that she needed to speak with him. without stopping, rivera told her that he did not have time to speak with her (his own supervisor) because he was working and drove by her."

        query: "Does ms. reed learning about the defaced guidelines precedes her advising ms. salome to issue the written discipline?"

        answer: "yes"
        """ 
    
example3 = f"""
        context: "rather than pursue the grievance through the contracts grievance and arbitration procedure, the union directed hospital employees to refuse to pick up extra shifts beyond their normal scheduled work. specifically, prior to the dispute over employees obligation to work holidays, employees would regularly and voluntarily pick up extra shifts beyond their normal scheduled work. but once the dispute over holiday work arose, employees stopped picking up shifts beyond their normal scheduled work. after the aforementioned dispute arose, a member of the union and a hospital employee provided to a member of the hospitals administration a text message stating, in part: per union unless you had originally picked up ot or st when it was first offered we should be giving back those hours for this coming week and they are asking us not to pick up anything through the end of may and the first 2 weeks of june because they still are not allowing people to take vacation on a holiday unless they have adequate staffing[.] upon information and belief, the references to ot and st in the text message mean overtime and straight time, respectively. the union members concerted refusal to pick up extra shifts is a violation of the contracts and has caused the hospital great hardship in achieving adequate staffing for the hospitals operations. as a result of the union and its members conduct, the hospital was forced to hire so-called traveler nurses and to pay an extraordinary premium, in the millions of case 3:23-cv-00466 document 1 filed 04/13/23 page 5 of 10 6 dollars, for adequate staffing compared with the cost to the hospital had the union not directed its members to engage in this concerted effort and had the union members picked up extra shifts as they had in the past."

        query: "Does the hospital being forced to hire traveler nurses and pay an extraordinary premium precedes the union members' concerted refusal to pick up extra shifts?”

        answer: "no"
        """
    
example1_cot = f"""
        context: "Before assigning the plaintiff to work, the defendant hired employees, including the plaintiff. The plaintiff was hired as a trackman."

        query: "Does hiring employees happens before assigning the plaintiff to work?"

        answer: "
        Reasoning:
        1. The defendant hired employees, including the plaintiff.
        2. The plaintiff was hired as a trackman.
        3. The employees including the plaintiff were all assigned to work.
        
        The correct answer is: "yes"
        """
    
example2_cot = f"""
        context: "ms. reed advised ms. salome to issue the written discipline and to speak with rivera (as well as the other roll tunnel operators) regarding the defaced guidelines. accordingly, ms. salome, along with clint kelch (the union steward) approached rivera to present him with the disciplinary warning and to discuss the defacement of the posted guidelines. rivera was on his clamp truck and ms. salome advised that rivera that she needed to speak with him. without stopping, rivera told her that he did not have time to speak with her (his own supervisor) because he was working and drove by her."

        query: "Does ms. reed learning about the defaced guidelines precedes her advising ms. salome to issue the written discipline?"

        answer: "
        Reasoning:

        1. Ms. Reed learned about the defaced guidelines.
        2. She then advised Ms. Salome to issue the written discipline.

        Since Ms. Reed's learning about the defaced guidelines happened before advising Ms. Salome, the correct answer is "yes".
        """  
    
example3_cot = f"""
        context: "rather than pursue the grievance through the contracts grievance and arbitration procedure, the union directed hospital employees to refuse to pick up extra shifts beyond their normal scheduled work. specifically, prior to the dispute over employees obligation to work holidays, employees would regularly and voluntarily pick up extra shifts beyond their normal scheduled work. but once the dispute over holiday work arose, employees stopped picking up shifts beyond their normal scheduled work. after the aforementioned dispute arose, a member of the union and a hospital employee provided to a member of the hospitals administration a text message stating, in part: per union unless you had originally picked up ot or st when it was first offered we should be giving back those hours for this coming week and they are asking us not to pick up anything through the end of may and the first 2 weeks of june because they still are not allowing people to take vacation on a holiday unless they have adequate staffing[.] upon information and belief, the references to ot and st in the text message mean overtime and straight time, respectively. the union members concerted refusal to pick up extra shifts is a violation of the contracts and has caused the hospital great hardship in achieving adequate staffing for the hospitals operations. as a result of the union and its members conduct, the hospital was forced to hire so-called traveler nurses and to pay an extraordinary premium, in the millions of case 3:23-cv-00466 document 1 filed 04/13/23 page 5 of 10 6 dollars, for adequate staffing compared with the cost to the hospital had the union not directed its members to engage in this concerted effort and had the union members picked up extra shifts as they had in the past."

        query: "Does the hospital being forced to hire traveler nurses and pay an extraordinary premium precedes the union members' concerted refusal to pick up extra shifts?"

        answer: "
        Reasoning:
        
        1. The union members' concerted refusal to pick up extra shifts occurred as a result of the dispute.
        2. This refusal led to the hospital being forced to hire traveler nurses and pay an extraordinary premium.

        Since the hospital was forced to hire traveler nurses and pay a premium as a result of the union members' refusal, the correct answer is "no".
        """





# Prompt templates
def get_base_prompt(context, query):
    return f"""
    Given the context, the task is to answer the query with "yes" or "no".

    context: "{context}"
    query: "{query}"
    answer:   
    """

def get_one_shot_prompt(context, query):
    return f"""
    Given the context, the task is to answer the query with "yes" or "no".

    example: {example1}

    context: "{context}"
    query: "{query}"
    answer: 
    """

def get_few_shot_prompt(context, query):
    return f"""
    Given the context, the task is to answer the query with "yes" or "no".

    example 1: {example1}
    example 2: {example2}
    example 3: {example3}

    context: "{context}"
    query: "{query}"
    answer: 
    """

def get_chain_of_thought_one_shot_prompt(context, query):
    return f"""
    Given the context, the task is to answer the query with "yes" or "no". 

    Let's reason through it step-by-step:

    example: {example1_cot}

    context: "{context}"
    query: "{query}"
    answer: 
    """

def get_chain_of_thought_few_shot_prompt(context, query):
    return f"""
    Given the context, the task is to answer the query with "yes" or "no". 

    Let's reason through it step-by-step:

    example 1: {example1_cot}
    example 2: {example2_cot}
    example 3: {example3_cot}

    context: "{context}"
    query: "{query}"
    answer: 
    """