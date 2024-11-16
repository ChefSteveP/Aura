import curses  # terminal-based UI package
import time

def main(stdscr):  
    curses.curs_set(0)  # 0 hides the cursor, making the interface cleaner
    stdscr.keypad(True)  # Enable special keys (like arrow keys) to be recognized
    
    # Setup the menu
    menu = ["Start Game", "Exit"]
    current_row = 0
    # Setup color pairs for the menu
    curses.start_color()  
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Define color pair 1: green background, black text
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Define color pair 1: white background, black text
    

    # Displays the title Card
    def print_title():
        stdscr.clear()
        ascii_art = [
            "    ____        __              __       ",
            "   / __ \\____  / /_  ___  _____/ /_____ _",
            "  / /_/ / __ \\/ __ \\/ _ \\/ ___/ __/ __ `/",
            " / _, _/ /_/ / /_/ /  __/ /  / /_/ /_/ / ",
            "/_/ |_|\\____/_.___/\\___/_/   \\__/\\__,_/  "
        ]
         # Print each line of ASCII art
        for i, line in enumerate(ascii_art):
            stdscr.addstr(i, 2, line, curses.color_pair(2))  # i + 2 gives a little margin from the top
        stdscr.addstr(len(ascii_art), 3, "Welcome to a new world of adventure!", curses.color_pair(2))
        
    # Function to display the menu with the selected option highlighted
    def print_menu(stdscr, selected_row): 
     
        for idx, row in enumerate(menu):
            # Highlight the selected row using a color pair
            if idx == selected_row:
                stdscr.addstr(idx + 8, 0, f"> {row} <", curses.color_pair(1))
                stdscr.addstr(idx + 8, len(row) + 4, "                 ", curses.color_pair(2))
            else:
                stdscr.addstr(idx + 8, 0, row + "                      ", curses.color_pair(2))  # Display other rows normally
        stdscr.refresh()  # Refresh the screen to show updated content
    
    # Function to return to the main menu
    def return_to_menu():
        stdscr.clear()
        print_title()
        print_menu(stdscr, 0)

    # Function to display a horizontal menu with blinking
    def horizontal_menu_with_blink(stdscr, text, options):
        stdscr.clear()
        options.append("Exit Game")
        # Display the story text in green
        for idx, line in enumerate(text.split("\n")):
            stdscr.addstr(idx + 2, 2, line, curses.color_pair(1))

        current_option = 0
        blink = True  # Toggle for blinking

        while True:
            # Render options
            option_x = 5
            for idx, option in enumerate(options):
                if idx == current_option:
                    # Use blinking attribute
                    if blink:
                        stdscr.addstr(10, option_x, f"[{option}]", curses.color_pair(2) | curses.A_BLINK)
                    else:
                        stdscr.addstr(10, option_x, f"[{option}]", curses.color_pair(2))
                else:
                    stdscr.addstr(10, option_x, option, curses.color_pair(1))
                option_x += len(option) + 5  # Add spacing between options
                
            # Toggle blink state
            blink = not blink
            time.sleep(0.3)  # Delay to control blink speed

            stdscr.refresh()
            key = stdscr.getch()

            if key == curses.KEY_LEFT and current_option > 0:
                current_option -= 1
            elif key == curses.KEY_RIGHT and current_option < len(options) - 1:
                current_option += 1
            elif key == ord("\n"):  # Enter key
                if options[current_row] == "Exit Game":
                    stdscr.refresh()  # Refresh the screen to show the message
                    stdscr.getch()  # Wait for the user to press a key before continuing
                    return_to_menu()
                    break
                return current_option  # Return the selected option index

            # Toggle blink state
            blink = not blink
            time.sleep(0.3)  # Delay to control blink speed

    

    print_title()
    # Display the initial menu
    print_menu(stdscr, current_row)

    # Main loop to handle user input and navigation
    while True:
        key = stdscr.getch()  # Wait for user input (key press)

        # Navigate up in the menu, ensuring the selection doesn't go out of bounds
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        # Navigate down in the menu, ensuring the selection doesn't go out of bounds
        elif key == curses.KEY_DOWN and current_row < len(menu) - 1:
            current_row += 1
        # Handle the Enter key to confirm a selection
        elif key == ord("\n"):  # ord("\n") detects the Enter key
            # Display the selected option at the bottom of the menu
            stdscr.addstr(len(menu) + 4, 2, f"You selected '{menu[current_row]}'", curses.color_pair(1))
            stdscr.refresh()  # Refresh the screen to show the message
            stdscr.getch()  # Wait for the user to press a key before continuing

            # If the "Exit" option is selected, break the loop and exit
            if menu[current_row] == "Start Game":
                 # Main flow
                story_text = "You find yourself in a dark forest. The trees tower above you, their branches like claws in the moonlight.\nWhat will you do?"
                options = ["Explore", "Run Away"]
                selected_option = horizontal_menu_with_blink(stdscr, story_text, options)
            if menu[current_row] == "Exit":
                break

        # Update the menu display with the new selected row
        print_menu(stdscr, current_row)

# The curses wrapper ensures proper initialization and cleanup of the terminal
curses.wrapper(main)
