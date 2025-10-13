from src.update_files import update_data, update_date

def start():
    update_date.update()
    choice = input("Would you like to update the data? (y/n): ")
    if choice.lower() == 'y':
        update_data.update()