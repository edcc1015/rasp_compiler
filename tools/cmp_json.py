import json
import sys


def normalize(obj):
    """Normalize JSON for semantic comparison.

    Removes empty ``"attrs"`` objects from Call nodes so that
    ``{"attrs": {}}`` and no ``"attrs"`` key compare as equal.
    """
    if isinstance(obj, dict):
        if obj.get("kind") == "Call" and obj.get("attrs") == {}:
            obj = {k: v for k, v in obj.items() if k != "attrs"}
        return {k: normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    return obj


def compare_json_simple(file1, file2):
    try:
        with open(file1, 'r', encoding='utf-8') as f1:
            json1 = normalize(json.load(f1))

        with open(file2, 'r', encoding='utf-8') as f2:
            json2 = normalize(json.load(f2))

        str1 = json.dumps(json1, sort_keys=True, indent=2)
        str2 = json.dumps(json2, sort_keys=True, indent=2)

        if str1 == str2:
            print("JSON files are identical!")
            return True
        else:
            print("JSON files are different!")

            lines1 = str1.splitlines()
            lines2 = str2.splitlines()

            for i, (line1, line2) in enumerate(zip(lines1, lines2)):
                if line1 != line2:
                    print(f"\n第一个差异位置 (行 {i+1}):")
                    print(f"文件1: {line1}")
                    print(f"文件2: {line2}")
                    break

            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_json.py <File 1> <File 2>")
        sys.exit(1)

    result = compare_json_simple(sys.argv[1], sys.argv[2])
    sys.exit(0 if result else 1)