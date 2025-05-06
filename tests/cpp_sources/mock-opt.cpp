// Mock opt for testing.

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  std::string input_path;
  std::string output_path;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "-o" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg[0] != '-') {
      input_path = arg;
    }
  }

  std::ifstream input(input_path);
  std::ofstream output(output_path);

  if (!input || !output) {
    std::cerr << "Failed to open input or output file\n";
    return 1;
  }

  std::string line;
  while (std::getline(input, line)) {
    output << line << "\n";
  }

  output << "\ntest mock_opt 42\n";

  return 0;
}
