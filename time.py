from datetime import datetime

current_time = datetime.now()
formatted_date = current_time.strftime("%d-%m-%Y")

print(current_time.strftime("%d-%m-%Y"))
print(current_time.strftime("%H:%M:%S"))
