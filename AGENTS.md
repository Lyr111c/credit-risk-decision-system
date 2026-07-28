# 项目代理说明

## 适用范围

本文件适用于整个仓库。更深层目录中的 `AGENTS.md` 可以为其目录补充更具体的要求，但不得降低本文件规定的双语交付标准。

## 面向读者的内容必须双语

- 所有新生成或实质性更新、供人阅读的报告和介绍性内容都必须同时提供简体中文和英文版本。
- 适用内容包括但不限于：README、项目或数据集介绍、数据质量报告、EDA 报告、实验总结、模型报告、模型卡、业务说明和交付说明。
- 中英文内容应放在同一个文件中，先给出完整中文版，再给出完整英文版。使用清晰的一级或二级标题分隔两种语言。
- 两个版本必须包含相同的事实、指标、结论、限制和警告。不得只翻译标题或摘要，也不得让其中一种语言明显简略。
- 代码标识符、变量名、文件路径、命令、公式和标准指标名称可以保留原文；其周围的解释、表头、图题和必要注释应提供对应语言版本。
- 对自动生成的报告，必须同时更新报告生成逻辑和相关测试，再重新生成报告。不要只手工修改生成产物，因为后续运行可能覆盖修改。
- 除非用户明确要求拆分文件，否则不要分别创建中文文件和英文文件。

## 不要求双语的内容

- 源代码、测试代码、配置文件、机器可读数据、JSON、日志和原始命令输出不要求翻译。
- 仅供程序内部使用且不会作为读者交付物的临时文件不要求双语。
- Git 提交信息保持简洁，可使用仓库当前惯用语言，不要求中英双语。

## 完成前检查

- 确认中文部分位于英文部分之前。
- 确认两个版本的数字、表格、结论和限制一致。
- 如果内容由代码生成，运行生成命令和相关测试，确认重新生成后仍保持双语。

---

# Project Agent Instructions

## Scope

This file applies to the entire repository. A deeper `AGENTS.md` may add more specific instructions for its directory, but it must not weaken the bilingual-delivery requirements defined here.

## Bilingual human-facing content

- Every newly generated or materially updated report or introductory document intended for human readers must be provided in both Simplified Chinese and English.
- This includes, but is not limited to, READMEs, project or dataset introductions, data-quality reports, EDA reports, experiment summaries, model reports, model cards, business explanations, and delivery notes.
- Put both languages in the same file. Present the complete Chinese version first and the complete English version second, separated by clear level-one or level-two headings.
- The two versions must contain the same facts, metrics, conclusions, limitations, and warnings. Translating only headings or summaries, or making one language materially less complete, is not acceptable.
- Code identifiers, variable names, file paths, commands, formulas, and standard metric names may remain unchanged. Provide corresponding translations for the surrounding explanations, table headings, figure captions, and necessary notes.
- For generated reports, update the report generator and relevant tests, then regenerate the report. Do not edit only the generated artifact, because a later run may overwrite the bilingual content.
- Do not create separate Chinese and English files unless the user explicitly requests separate files.

## Content that does not require translation

- Source code, test code, configuration, machine-readable data, JSON, logs, and raw command output do not require translation.
- Temporary files used only by programs and not delivered to readers do not require translation.
- Git commit messages should remain concise and may follow the repository's current language convention; they do not need to be bilingual.

## Pre-completion checks

- Confirm that the Chinese section appears before the English section.
- Confirm that numbers, tables, conclusions, and limitations agree across both versions.
- If code generates the content, run the generation command and relevant tests to confirm that regenerated output remains bilingual.
