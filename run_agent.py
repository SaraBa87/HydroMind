import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mutil_tool_agent.agent import root_agent

if __name__ == "__main__":
    # Your agent execution code here
    print("Agent initialized:", root_agent) 