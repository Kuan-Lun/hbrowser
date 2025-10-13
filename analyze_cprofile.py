#!/usr/bin/env python3
"""
cProfile結果分析工具

這個腳本會解析cProfile的文本輸出，並以不同的排序方式顯示結果，
幫助分析效能瓶頸。
"""

import re
import argparse
from typing import List


class ProfileLine:
    def __init__(
        self,
        ncalls: int,
        tottime: float,
        percall_tot: float,
        cumtime: float,
        percall_cum: float,
        filename_function: str,
    ):
        self.ncalls = ncalls
        self.tottime = tottime
        self.percall_tot = percall_tot
        self.cumtime = cumtime
        self.percall_cum = percall_cum
        self.filename_function = filename_function

        # 解析模組和函數名
        self.module_name = self._extract_module_name()
        self.function_name = self._extract_function_name()

    def _extract_module_name(self) -> str:
        """提取模組名稱"""
        if "(" in self.filename_function:
            filepath = self.filename_function.split("(")[0]
            if "/" in filepath or "\\" in filepath:
                # 取檔案名稱
                return filepath.split("/")[-1].split("\\")[-1]
            else:
                return filepath
        return "unknown"

    def _extract_function_name(self) -> str:
        """提取函數名稱"""
        if "(" in self.filename_function and ")" in self.filename_function:
            return self.filename_function.split("(")[-1].rstrip(")")
        return "unknown"

    def is_your_code(self) -> bool:
        """判斷是否為用戶自己的代碼"""
        user_modules = ["hv.py", "gallery.py", "hv_battle", "main.py"]
        return any(module in self.filename_function for module in user_modules)

    def is_hbrowser_package(self) -> bool:
        """判斷是否為 hbrowser 套件的代碼"""
        hbrowser_indicators = [
            "hbrowser/",
            "hvbrowser/",
            "\\hbrowser\\",
            "\\hvbrowser\\",
            "src/hbrowser",
            "src/hvbrowser",
            "src\\hbrowser",
            "src\\hvbrowser",
            "gallery.py",  # 直接的檔案名稱
            "hv.py",  # 直接的檔案名稱
        ]
        return any(
            indicator in self.filename_function for indicator in hbrowser_indicators
        )

    def get_hbrowser_module(self) -> str:
        """取得 hbrowser 套件中的具體模組名稱"""
        if not self.is_hbrowser_package():
            return "non-hbrowser"

        # 檢查直接的檔案名稱
        if "gallery.py" in self.filename_function:
            return "hbrowser.gallery"
        elif "hv.py" in self.filename_function:
            return "hvbrowser.hv"

        # 提取模組路徑
        if (
            "hbrowser/" in self.filename_function
            or "hbrowser\\" in self.filename_function
        ):
            parts = self.filename_function.split("hbrowser")[-1].strip("/\\")
            if "/" in parts or "\\" in parts:
                return "hbrowser." + parts.split("/")[0].split("\\")[0].replace(
                    ".py", ""
                )
            else:
                return "hbrowser.main"
        elif (
            "hvbrowser/" in self.filename_function
            or "hvbrowser\\" in self.filename_function
        ):
            parts = self.filename_function.split("hvbrowser")[-1].strip("/\\")
            if "/" in parts or "\\" in parts:
                return "hvbrowser." + parts.split("/")[0].split("\\")[0].replace(
                    ".py", ""
                )
            else:
                return "hvbrowser.main"
        return "hbrowser.unknown"

    def is_selenium(self) -> bool:
        """判斷是否為selenium相關"""
        selenium_keywords = ["selenium", "webdriver", "action_chains", "action_builder"]
        return any(
            keyword in self.filename_function.lower() for keyword in selenium_keywords
        )

    def is_network(self) -> bool:
        """判斷是否為網路相關"""
        network_keywords = [
            "client.py",
            "connection",
            "socket",
            "ssl",
            "_request_methods",
        ]
        return any(
            keyword in self.filename_function.lower() for keyword in network_keywords
        )

    def is_html_parsing(self) -> bool:
        """判斷是否為HTML解析相關"""
        html_keywords = ["_htmlparser", "beautifulsoup", "feedparser", "element.py"]
        return any(
            keyword in self.filename_function.lower() for keyword in html_keywords
        )


class CProfileAnalyzer:
    def __init__(self, filename: str):
        self.filename = filename
        self.lines: List[ProfileLine] = []
        self.total_time = 0.0

    def parse_file(self):
        """解析cProfile文件"""
        with open(self.filename, "r", encoding="utf-8") as f:
            content = f.read()

        # 提取總時間
        time_match = re.search(r"(\d+) function calls.*in ([\d.]+) seconds", content)
        if time_match:
            self.total_time = float(time_match.group(2))

        # 解析每一行
        lines = content.split("\n")
        for line in lines:
            # 匹配profile行的正則表達式
            match = re.match(
                r"\s*(\d+(?:/\d+)?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)",
                line,
            )
            if match:
                ncalls_str = match.group(1)
                ncalls = (
                    int(ncalls_str.split("/")[0])
                    if "/" in ncalls_str
                    else int(ncalls_str)
                )

                profile_line = ProfileLine(
                    ncalls=ncalls,
                    tottime=float(match.group(2)),
                    percall_tot=float(match.group(3)),
                    cumtime=float(match.group(4)),
                    percall_cum=float(match.group(5)),
                    filename_function=match.group(6),
                )
                self.lines.append(profile_line)

    def analyze_by_cumtime(self, top_n: int = 20):
        """按累計時間排序分析"""
        print(f"\n=== 前 {top_n} 名最耗時函數 (按累計時間排序) ===")
        print(
            f"{'累計時間':<10} {'自身時間':<10} {'調用次數':<10} {'模組':<20} {'函數':<30}"
        )
        print("-" * 90)

        sorted_lines = sorted(self.lines, key=lambda x: x.cumtime, reverse=True)
        for line in sorted_lines[:top_n]:
            print(
                f"{line.cumtime:<10.3f} {line.tottime:<10.3f} {line.ncalls:<10} "
                f"{line.module_name[:19]:<20} {line.function_name[:29]:<30}"
            )

    def analyze_by_tottime(self, top_n: int = 20):
        """按自身時間排序分析"""
        print(f"\n=== 前 {top_n} 名最耗時函數 (按自身時間排序) ===")
        print(
            f"{'自身時間':<10} {'累計時間':<10} {'調用次數':<10} {'模組':<20} {'函數':<30}"
        )
        print("-" * 90)

        sorted_lines = sorted(self.lines, key=lambda x: x.tottime, reverse=True)
        for line in sorted_lines[:top_n]:
            print(
                f"{line.tottime:<10.3f} {line.cumtime:<10.3f} {line.ncalls:<10} "
                f"{line.module_name[:19]:<20} {line.function_name[:29]:<30}"
            )

    def analyze_user_code(self, top_n: int = 10):
        """分析用戶自己的代碼"""
        print(f"\n=== 用戶代碼效能分析 (前 {top_n} 名) ===")
        print(f"{'累計時間':<10} {'自身時間':<10} {'調用次數':<10} {'函數位置':<50}")
        print("-" * 90)

        user_lines = [line for line in self.lines if line.is_your_code()]
        sorted_lines = sorted(user_lines, key=lambda x: x.cumtime, reverse=True)

        for line in sorted_lines[:top_n]:
            print(
                f"{line.cumtime:<10.3f} {line.tottime:<10.3f} {line.ncalls:<10} "
                f"{line.filename_function[:49]:<50}"
            )

    def analyze_hbrowser_package(self, top_n: int = 20):
        """詳細分析 hbrowser 套件效能"""
        print(f"\n=== hbrowser 套件效能分析 (前 {top_n} 名) ===")
        print(
            f"{'累計時間':<10} {'自身時間':<10} {'調用次數':<10} {'模組':<25} {'函數':<30}"
        )
        print("-" * 95)

        hbrowser_lines = [line for line in self.lines if line.is_hbrowser_package()]
        sorted_lines = sorted(hbrowser_lines, key=lambda x: x.cumtime, reverse=True)

        for line in sorted_lines[:top_n]:
            module = line.get_hbrowser_module()
            print(
                f"{line.cumtime:<10.3f} {line.tottime:<10.3f} {line.ncalls:<10} "
                f"{module[:24]:<25} {line.function_name[:29]:<30}"
            )

        # 特別分析 battle_in_turn 相關的函數調用
        self.analyze_battle_in_turn_details(hbrowser_lines)

        if not hbrowser_lines:
            print("未找到 hbrowser 套件相關的函數調用")
            return

        # 分析真正的瓶頸：自身時間少但累計時間多的函數（說明大部分時間花在調用其他函數上）
        print("\n=== hbrowser 真正效能瓶頸 (累計時間高但自身時間低的函數) ===")
        print(
            f"{'累計時間':<10} {'自身時間':<10} {'時間比率':<10} {'調用次數':<10} {'函數位置':<50}"
        )
        print("-" * 100)

        bottleneck_candidates = []
        for line in hbrowser_lines:
            if line.cumtime > 0.1:  # 累計時間超過0.1秒
                # 計算時間比率：累計時間 / 自身時間，比率越高說明越多時間花在調用其他函數
                ratio = (
                    line.cumtime / line.tottime
                    if line.tottime > 0.001
                    else float("inf")
                )
                if ratio > 10:  # 累計時間是自身時間的10倍以上
                    bottleneck_candidates.append({"line": line, "ratio": ratio})

        # 按累計時間排序
        bottleneck_candidates.sort(key=lambda x: x["line"].cumtime, reverse=True)

        for candidate in bottleneck_candidates[:15]:
            line = candidate["line"]
            ratio = candidate["ratio"]
            ratio_str = f"{ratio:.1f}x" if ratio != float("inf") else "∞"
            print(
                f"{line.cumtime:<10.3f} {line.tottime:<10.3f} {ratio_str:<10} {line.ncalls:<10} "
                f"{line.filename_function[:49]:<50}"
            )

        # 按模組分組統計
        print("\n=== hbrowser 套件按模組分組統計 ===")
        module_stats = {}
        for line in hbrowser_lines:
            module = line.get_hbrowser_module()
            if module not in module_stats:
                module_stats[module] = {"cumtime": 0.0, "tottime": 0.0, "count": 0}
            module_stats[module]["cumtime"] += line.cumtime
            module_stats[module]["tottime"] += line.tottime
            module_stats[module]["count"] += 1

        print(f"{'模組':<30} {'累計時間':<12} {'自身時間':<12} {'函數數量':<10}")
        print("-" * 70)

        for module, stats in sorted(
            module_stats.items(), key=lambda x: x[1]["cumtime"], reverse=True
        ):
            print(
                f"{module[:29]:<30} {stats['cumtime']:<12.3f} {stats['tottime']:<12.3f} {stats['count']:<10}"
            )

        # 找出最可能的效能瓶頸
        print("\n=== hbrowser 效能瓶頸分析 ===")
        print(
            "這些函數累計時間長，但自身時間短，說明大部分時間花在調用其他函數（如 selenium）:"
        )

        if bottleneck_candidates:
            for i, candidate in enumerate(bottleneck_candidates[:10], 1):
                line = candidate["line"]
                ratio = candidate["ratio"]
                print(f"\n{i}. {line.get_hbrowser_module()}.{line.function_name}")
                print(f"   累計時間: {line.cumtime:.3f}秒 (包含調用其他函數的時間)")
                print(f"   自身時間: {line.tottime:.3f}秒 (函數本身的執行時間)")
                print(f"   時間比率: {ratio:.1f}x (累計/自身)")
                print(f"   調用次數: {line.ncalls}")
                print(f"   位置: {line.filename_function}")
                print("   → 這個函數可能調用了耗時的 selenium 或網路操作")
        else:
            print("未檢測到明顯的 hbrowser 效能瓶頸")

    def analyze_external_calls_impact(self):
        """分析外部函數調用對 hbrowser 的影響"""
        print("\n=== 外部函數對 hbrowser 的效能影響分析 ===")

        # 找出最耗時的外部函數
        external_lines = [line for line in self.lines if not line.is_hbrowser_package()]
        top_external = sorted(external_lines, key=lambda x: x.cumtime, reverse=True)[
            :10
        ]

        # 找出最耗時的 hbrowser 函數
        hbrowser_lines = [line for line in self.lines if line.is_hbrowser_package()]
        top_hbrowser = sorted(hbrowser_lines, key=lambda x: x.cumtime, reverse=True)[
            :10
        ]

        print("最耗時的外部函數:")
        print(f"{'累計時間':<10} {'自身時間':<10} {'調用次數':<10} {'函數':<50}")
        print("-" * 90)

        for line in top_external:
            print(
                f"{line.cumtime:<10.3f} {line.tottime:<10.3f} {line.ncalls:<10} {line.filename_function[:49]:<50}"
            )

        print("\n對應的 hbrowser 函數 (可能調用了上述外部函數):")
        print(
            f"{'累計時間':<10} {'自身時間':<10} {'時間比率':<10} {'調用次數':<10} {'函數':<40}"
        )
        print("-" * 95)

        for line in top_hbrowser:
            if line.cumtime > 0.1:
                ratio = (
                    line.cumtime / line.tottime
                    if line.tottime > 0.001
                    else float("inf")
                )
                if ratio > 5:  # 只顯示可能調用外部函數的 hbrowser 函數
                    ratio_str = f"{ratio:.1f}x" if ratio != float("inf") else "∞"
                    print(
                        f"{line.cumtime:<10.3f} {line.tottime:<10.3f} {ratio_str:<10} {line.ncalls:<10} {line.function_name[:39]:<40}"
                    )

        print("\n💡 分析建議:")
        print(
            "- 累計時間高但自身時間低的 hbrowser 函數，表示大部分時間花在調用外部函數"
        )
        print("- 重點優化這些 hbrowser 函數中對 selenium/網路操作的調用方式")
        print("- 考慮減少不必要的 find_element 調用或增加等待策略優化")

    def analyze_time_distribution(self):
        """分析時間分布和調用關係"""
        print("\n=== 時間分布分析 ===")

        # 計算真正消耗時間的函數（自身時間較高的）
        high_tottime = [line for line in self.lines if line.tottime > 0.1]
        high_tottime.sort(key=lambda x: x.tottime, reverse=True)

        print("🔥 真正耗時的函數 (自身時間 > 0.1秒):")
        print(
            f"{'自身時間':<10} {'累計時間':<10} {'時間放大':<10} {'調用次數':<10} {'函數':<40}"
        )
        print("-" * 85)

        for line in high_tottime[:15]:
            amplification = line.cumtime / line.tottime if line.tottime > 0 else 0
            print(
                f"{line.tottime:<10.3f} {line.cumtime:<10.3f} {amplification:<10.1f}x {line.ncalls:<10} {line.function_name[:39]:<40}"
            )

        # 分析調用放大效應最嚴重的函數
        high_amplification = []
        for line in self.lines:
            if line.tottime > 0.01 and line.cumtime > 0.1:
                amplification = line.cumtime / line.tottime
                if amplification > 20:  # 累計時間是自身時間的20倍以上
                    high_amplification.append((line, amplification))

        high_amplification.sort(key=lambda x: x[1], reverse=True)

        print("\n🔄 調用放大效應最嚴重的函數 (可能是效能瓶頸的入口點):")
        print(
            f"{'放大倍數':<10} {'累計時間':<10} {'自身時間':<10} {'調用次數':<10} {'函數':<40}"
        )
        print("-" * 85)

        for line, amplification in high_amplification[:10]:
            print(
                f"{amplification:<10.1f}x {line.cumtime:<10.3f} {line.tottime:<10.3f} {line.ncalls:<10} {line.function_name[:39]:<40}"
            )

        if high_amplification:
            print("\n💡 分析建議:")
            print("• 放大倍數高的函數是調用鏈的上游，優化它們可能帶來最大收益")
            print("• 這些函數本身執行很快，但調用了很多耗時的子函數")
            print("• 考慮減少這些函數的調用次數，或優化其調用的子函數")

    def analyze_by_category(self):
        """按類別分析"""
        print("\n=== 按類別分析總耗時 ===")

        categories = {
            "hbrowser套件": lambda x: x.is_hbrowser_package(),
            "用戶代碼": lambda x: x.is_your_code() and not x.is_hbrowser_package(),
            "Selenium": lambda x: x.is_selenium(),
            "網路請求": lambda x: x.is_network(),
            "HTML解析": lambda x: x.is_html_parsing(),
            "其他": lambda x: not (
                x.is_hbrowser_package()
                or x.is_your_code()
                or x.is_selenium()
                or x.is_network()
                or x.is_html_parsing()
            ),
        }

        print(f"{'類別':<15} {'累計時間':<12} {'佔總時間%':<12} {'函數數量':<10}")
        print("-" * 60)

        for category_name, filter_func in categories.items():
            category_lines = [line for line in self.lines if filter_func(line)]
            total_cumtime = sum(line.cumtime for line in category_lines)
            percentage = (
                (total_cumtime / self.total_time * 100) if self.total_time > 0 else 0
            )
            count = len(category_lines)

            print(
                f"{category_name:<15} {total_cumtime:<12.3f} {percentage:<12.1f} {count:<10}"
            )

    def find_frequent_calls(self, min_calls: int = 1000):
        """尋找高頻調用的函數"""
        print(f"\n=== 高頻調用函數 (調用次數 >= {min_calls}) ===")
        print(
            f"{'調用次數':<12} {'累計時間':<10} {'平均時間':<12} {'模組':<20} {'函數':<30}"
        )
        print("-" * 95)

        frequent_lines = [line for line in self.lines if line.ncalls >= min_calls]
        sorted_lines = sorted(frequent_lines, key=lambda x: x.ncalls, reverse=True)

        for line in sorted_lines[:20]:
            avg_time = line.cumtime / line.ncalls if line.ncalls > 0 else 0
            print(
                f"{line.ncalls:<12} {line.cumtime:<10.3f} {avg_time:<12.6f} "
                f"{line.module_name[:19]:<20} {line.function_name[:29]:<30}"
            )

    def search_functions(self, keyword: str):
        """搜尋包含特定關鍵字的函數"""
        print(f"\n=== 搜尋結果: '{keyword}' ===")
        print(f"{'累計時間':<10} {'自身時間':<10} {'調用次數':<10} {'函數位置':<50}")
        print("-" * 90)

        matching_lines = [
            line
            for line in self.lines
            if keyword.lower() in line.filename_function.lower()
        ]
        sorted_lines = sorted(matching_lines, key=lambda x: x.cumtime, reverse=True)

        for line in sorted_lines:
            print(
                f"{line.cumtime:<10.3f} {line.tottime:<10.3f} {line.ncalls:<10} "
                f"{line.filename_function[:49]:<50}"
            )

    def performance_summary(self):
        """效能總結"""
        print("\n=== 效能分析總結 ===")
        print(f"總執行時間: {self.total_time:.3f} 秒")
        print(f"總函數調用數: {sum(line.ncalls for line in self.lines):,}")
        print(f"分析的函數數量: {len(self.lines)}")

        # 找出最耗時的前5個函數（按累計時間）
        top_5_cumtime = sorted(self.lines, key=lambda x: x.cumtime, reverse=True)[:5]
        print("\n最耗時的5個函數 (按累計時間 - 包含子函數調用):")
        print("💡 注意：累計時間可能超過總執行時間，因為包含重疊的子函數調用時間")
        for i, line in enumerate(top_5_cumtime, 1):
            percentage = (
                (line.cumtime / self.total_time * 100) if self.total_time > 0 else 0
            )
            print(
                f"  {i}. {line.function_name} - {line.cumtime:.3f}秒 ({percentage:.1f}%)"
            )

        # 找出最耗時的前5個函數（按自身時間）
        top_5_tottime = sorted(self.lines, key=lambda x: x.tottime, reverse=True)[:5]
        print("\n最耗時的5個函數 (按自身時間 - 不包含子函數調用):")
        total_tottime = sum(line.tottime for line in self.lines)
        for i, line in enumerate(top_5_tottime, 1):
            percentage = (
                (line.tottime / total_tottime * 100) if total_tottime > 0 else 0
            )
            print(
                f"  {i}. {line.function_name} - {line.tottime:.3f}秒 ({percentage:.1f}%)"
            )

        print("\n📊 時間分析說明:")
        print("• 累計時間 (cumtime): 包含該函數及其所有子函數的執行時間")
        print("• 自身時間 (tottime): 僅該函數本身的執行時間（不含子函數）")
        print(f"• 累計時間總和: {sum(line.cumtime for line in self.lines):.3f}秒")
        print(f"• 自身時間總和: {total_tottime:.3f}秒")
        print(f"• 實際執行時間: {self.total_time:.3f}秒")
        print("• 累計時間超過實際時間是正常的，因為子函數調用時間被重複計算")

    def analyze_battle_in_turn_details(self, hbrowser_lines):
        """詳細分析 battle_in_turn 內部的函數調用"""
        print("\n=== battle_in_turn 內部函數調用分析 ===")

        # 找到 battle_in_turn 函數
        battle_in_turn_line = None
        for line in hbrowser_lines:
            if "battle_in_turn" in line.function_name:
                battle_in_turn_line = line
                break

        if not battle_in_turn_line:
            print("未找到 battle_in_turn 函數")
            return

        print("battle_in_turn 總計:")
        print(f"  累計時間: {battle_in_turn_line.cumtime:.3f}秒")
        print(f"  自身時間: {battle_in_turn_line.tottime:.3f}秒")
        print(f"  調用次數: {battle_in_turn_line.ncalls}")
        print(
            f"  平均每次調用: {battle_in_turn_line.cumtime/battle_in_turn_line.ncalls:.3f}秒"
        )

        # 分析 battle_in_turn 中可能調用的函數
        battle_related_functions = [
            "finish_battle",
            "go_next_floor",
            "check_hp",
            "check_mp",
            "check_sp",
            "check_overcharge",
            "apply_buff",
            "use_channeling",
            "attack",
            "get_stat_percent",
            "PonyChart",
            "get_alive_monsters_elements",
            "alive_system_monster_ids",
            "get_monster_ids_with_debuff",
            "_is_monster_alive",
            "click_and_wait_log",
            "find_element",
        ]

        print("\n可能由 battle_in_turn 調用的函數 (按累計時間排序):")
        print(
            f"{'累計時間':<10} {'自身時間':<10} {'調用次數':<10} {'平均耗時':<10} {'函數':<30}"
        )
        print("-" * 85)

        related_lines = []
        for line in self.lines:
            for func_name in battle_related_functions:
                if func_name.lower() in line.function_name.lower():
                    avg_time = line.cumtime / line.ncalls if line.ncalls > 0 else 0
                    related_lines.append((line, avg_time))
                    break

        # 按累計時間排序
        related_lines.sort(key=lambda x: x[0].cumtime, reverse=True)

        for line, avg_time in related_lines[:20]:
            print(
                f"{line.cumtime:<10.3f} {line.tottime:<10.3f} {line.ncalls:<10} "
                f"{avg_time:<10.3f} {line.function_name[:29]:<30}"
            )

        # 分析平均每次調用最慢的函數
        print("\n平均每次調用最慢的函數:")
        print(f"{'平均耗時':<10} {'累計時間':<10} {'調用次數':<10} {'函數':<40}")
        print("-" * 75)

        slow_avg_functions = [(line, avg) for line, avg in related_lines if avg > 0.01]
        slow_avg_functions.sort(key=lambda x: x[1], reverse=True)

        for line, avg_time in slow_avg_functions[:15]:
            print(
                f"{avg_time:<10.3f} {line.cumtime:<10.3f} {line.ncalls:<10} {line.function_name[:39]:<40}"
            )


def main():
    parser = argparse.ArgumentParser(description="分析cProfile結果")
    parser.add_argument("filename", help="cProfile結果文件路徑")
    parser.add_argument(
        "--cumtime", "-c", type=int, default=20, help="顯示前N個最耗時函數(按累計時間)"
    )
    parser.add_argument(
        "--tottime", "-t", type=int, default=20, help="顯示前N個最耗時函數(按自身時間)"
    )
    parser.add_argument(
        "--user", "-u", type=int, default=10, help="顯示前N個用戶代碼函數"
    )
    parser.add_argument(
        "--frequent", "-f", type=int, default=1000, help="顯示調用次數超過N的函數"
    )
    parser.add_argument("--search", "-s", type=str, help="搜尋包含特定關鍵字的函數")
    parser.add_argument(
        "--hbrowser", "-b", action="store_true", help="詳細分析 hbrowser 套件效能"
    )

    args = parser.parse_args()

    analyzer = CProfileAnalyzer(args.filename)
    analyzer.parse_file()

    # 執行各種分析
    analyzer.performance_summary()

    if args.hbrowser:
        # 如果指定了 hbrowser 參數，重點分析 hbrowser
        analyzer.analyze_hbrowser_package(30)
        analyzer.analyze_external_calls_impact()  # 新增外部調用影響分析
        analyzer.search_functions("hbrowser")
        analyzer.search_functions("hvbrowser")
    else:
        analyzer.analyze_by_cumtime(args.cumtime)
        analyzer.analyze_by_tottime(args.tottime)
        analyzer.analyze_hbrowser_package(20)  # 仍然包含 hbrowser 分析
        analyzer.analyze_user_code(args.user)
        analyzer.analyze_by_category()
        analyzer.find_frequent_calls(args.frequent)

    if args.search:
        analyzer.search_functions(args.search)


if __name__ == "__main__":
    # 如果沒有命令行參數，使用默認文件
    import sys

    if len(sys.argv) == 1:
        # 默認分析當前目錄的cProfile_result.txt
        analyzer = CProfileAnalyzer("cProfile_result.txt")
        analyzer.parse_file()

        analyzer.performance_summary()
        analyzer.analyze_time_distribution()  # 新增時間分布分析
        analyzer.analyze_by_cumtime(15)
        analyzer.analyze_by_tottime(15)
        analyzer.analyze_hbrowser_package(20)  # 新增 hbrowser 分析
        analyzer.analyze_user_code(10)
        analyzer.analyze_by_category()
        analyzer.find_frequent_calls(1000)

        # 針對常見問題進行特定搜索
        print("\n" + "=" * 80)
        analyzer.search_functions("hbrowser")  # 搜索 hbrowser 相關
        analyzer.search_functions("hvbrowser")  # 搜索 hvbrowser 相關
        analyzer.search_functions("selenium")
        analyzer.search_functions("wait")
        analyzer.search_functions("find_element")
    else:
        main()
