import random


# A mock function that simulates an external API call (like Jira or ServiceNow)
def create_it_ticket(issue_details: str):
    """
    Use this tool when a user wants to report a technical issue,
    reset a password, or ask for help from IT support.
    """
    ticket_id = f"TKT-{random.randint(1000, 9999)}"
    print(f"\n[SYSTEM] Creating ticket for: {issue_details}")
    print(f"[SYSTEM] Ticket assigned ID: {ticket_id}\n")

    return f"Ticket created successfully! Your Ticket ID is {ticket_id}. Support will contact you shortly."
