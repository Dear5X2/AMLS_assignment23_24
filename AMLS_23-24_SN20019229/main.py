import subprocess


# Function to run a Python script using subprocess
def run_script(script_path):
    subprocess.run(["python", script_path], check=True)

if __name__ == "__main__":
    # Ask the user which script to run
    choice = input("Which script would you like to run? (1 for Script in A, 2 for Script in B): ")
    
    if choice == '1':
        # Run the script in folder A
        run_script('A/PneumoniaMNIST.py')  # Replace script_name.py with your actual script name
    elif choice == '2':
        # Run the script in folder B
        run_script('B/PathMNIST.py')  # Replace script_name.py with your actual script name
    else:
        print("Invalid selection.")
