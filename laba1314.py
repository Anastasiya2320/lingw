with open("C:Users\ноут\Desktop\lingv\mymy.txt", encoding="utf-8") as file:
    c = file.read()
    tokens = c.split(" ")
    print(tokens)