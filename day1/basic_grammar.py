name = "xv6"

topics = ["syscall", "process", "pipe", "filesystem"]

# for topic in topics:
#     print(f"I am learning {topic} in {name}")

def summarize_note(title, topics):
    return{
        "title" : title,
        "topic_count" : len(topics),
        "topics" : topics
    }

result = summarize_note("OS Notes", ["process", "syscall", "pipe"])
print(result)