import pandas as pd
import os
import re

if __name__ == "__main__":
    part = """
Here are my answers to the 10 questions:

Q1: I would score a 4 (Somewhat agree) for the statement "What each of us does in this situation affects the other." Based on the conversation, it is clear that Person PID_176 and I are interacting and that our actions have an impact on each other.

Q2: I would score a 2 (Somewhat disagree) for the statement "Our preferred outcomes in this situation are conflicting." While it is clear that Person PID_176 and I have different preferences (e.g., Person PID_176 wants to know how I'm doing, while I am focused on studying or working), it is not necessarily a conflicting outcome.

Q3: I would score a 4 (Somewhat agree) for the statement "How we behave now will have consequences for future outcomes." Based on the conversation, it is clear that our actions in this moment will impact our future interactions.

Q4: I would score a 3 (Neither person X nor myself) for the statement "We both know what the other wants." While it is clear that Person PID_176 is trying to initiate a conversation, it is not necessarily clear what they want or what I want.

Q5: I would score a 1 (Definitely myself) for the statement "Whatever each of us does in this situation, our actions will not affect the other's outcome." Based on the conversation, it is clear that my actions will impact Person PID_176's outcome, but not the other way around.

Q6: I would score a 4 (Somewhat agree) for the statement "We can both obtain our preferred outcomes." While it is clear that Person PID_176 has a different preference than I do, it is possible that we could both obtain our desired outcomes in this situation.

Q7: I would score a 3 (Neither person X nor myself) for the statement "Our future interactions are not affected by the outcomes of this situation." While it is clear that our actions in this moment will impact our future interactions, it is not necessarily clear how the outcome of this situation will affect future interactions.

Q8: I would score a 2
"""

    part = re.sub(r'Q\d+', '', part)
    numbers = re.findall(r'\b\d\b', part)
    scores = list(map(int, numbers))

    print(scores)