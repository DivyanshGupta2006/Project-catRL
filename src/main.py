from src.update_files import update_data

def start():
    choice = input("Would you like to update the data? (y/n): ")
    if choice.lower() == 'y':
        update_data.update()