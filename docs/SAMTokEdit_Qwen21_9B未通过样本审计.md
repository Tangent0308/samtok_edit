# Qwen3.5-9B noref 未通过样本审计

记录日期：2026-09-29。与[三模型对照实验](SAMTokEdit_Qwen21_noref提示词优化与模型复测.md)使用同一 prompt、温度 0、最多三次重试。本页引用的是逐条保存的原始两字段输出，mask 仅沿用数据集现有标注，不核对几何准确性。

## 1. 两种检查及其分母

程序检查在[validate_annotation](../samtok_edit21/annotate_full.py#L275)和[verify_semantic_review](../samtok_edit21/annotate_full.py#L322)内执行：ref 必须是原文唯一片段，ref 与 this region 数量对应，编辑类型/占位符符合协议，多实例只能绑定已有 mask ID，并检查若干明确的 OLD/NEW 词项约束。正常输出失败后最多重试三次；三次仍不合格才标记 `failed`。它**没有**逐条理解图像编辑语义。

我另对预先固定的 62 条对照样本与独立 60 条留出样本逐条阅读 `ref_phrase` 和 `noref_instruction`：审查旧定位是否消失、新内容与约束是否保留、引用对象/部件是否正确、是否把未编辑的比较对象当成额外 region。这是助手文本审阅，**不是独立人工金标，也不能代表全量准确率**。

| 范围 | 程序通过 | 程序拒绝 | 逐条文本审阅通过 | 两者同时通过 | 程序接受但审阅不通过 | 审阅通过但程序拒绝 |
|---|---:|---:|---:|---:|---:|---:|
| 312 条对照输入 | 270 | 42 | 固定 62 条中 51 | 62 条中 50 | 62 条中 5 | 62 条中 1 |
| 60 条留出输入 | 56 | 4 | 53 | 52 | 4 | 1 |
| 合计 372 输入 / 122 审阅 | 326 | 46 | 104/122 | 102/122 | 9/122 | 2/122 |

因此“9B 不通过”有两种不同含义：**46 条程序 rejected**；固定审阅里另有**9 条程序 accepted 但语义错误**。两类不能相加当全量失败率，因为只有 122 条做了人工式文本审阅。

## 2. 程序拒绝的 46 条

| 最终错误归因 | 数量 | 含义 |
|---|---:|---|
| 已有多实例 mask 无法唯一绑定 | 18 | 输出多个编辑单元，但根据已有实例描述无法唯一绑定现成 mask；不意味着数据集 mask 像素有错。 |
| add 引用遗漏新增内容 | 7 | ref 只选择已有承载物/位置，未包括新加的物体及位置。 |
| ref 与 region 数量不一致 | 7 | 列表中的编辑引用数量与 noref 的 region 占位数量不同。 |
| ref 不是原文唯一片段 | 4 | 模型拼接、改写或省略原文，导致片段不能唯一精确匹配。 |
| noref 丢原文词项（可能误拒） | 3 | 原词丢失；其中同义替换或代词省略可能是程序过严。 |
| 其他（文字或部件规则） | 3 | OLD 文字、引用部件或文本类型专门校验失败。 |
| 协议语法拒绝 | 2 | 插入 mask 占位符后不符合协议句法。 |
| 粗类型无法可靠细化 | 2 | 数据原生粗类别无法安全映射到本协议原子类型。 |

拒绝主要集中在 ScaleEdit（36 条），另有 CrispEdit 9 条、RefEdit 1 条；Derived 在这 372 条中程序拒绝 0 条。样本是分层抽取并含历史难例，不能据此推断全量数据集失败率。下面列出每一条最终错误，完整最后一次输出也保留在 `/tmp/samtok21-qwen35-compare/all_program_failures.jsonl`。

| 数据集 / ID | 原始 instruction | 三次尝试后的最终错误 |
|---|---|---|
| crispedit / `crispedit-7e5d9977e38dc89b0db0373d` | The person lifts the kitten closer to their face. | An operation has no unambiguously matched existing mask |
| crispedit / `crispedit-e9499b26467e5d81ad109ce4` | A man lowers a bottle of wine back onto the table. | Ambiguous existing-mask binding for source_0 |
| refedit / `refedit-0caa4170b27a8ea3704c9db1` | Introduce a plain birdbath near the rose bushes in the garden | Missing original words: introduce. Keep old locating details in ref_phrase and new content/constraints in noref_instruction |
| scaleedit / `scaleedit-5505b1c66abb2645c581e3a1` | Spread the wings of the angel statue on the left side of the bridge and extend its right arm outward. | An operation has no unambiguously matched existing mask |
| scaleedit / `scaleedit-6b077d441551e3571e8bcea3` | Change the color of the large golden Buddha statue, the smaller golden Buddha statue in the foreground, the ceiling fixture, and the halo to silver. | Ambiguous existing-mask binding for unit_4_4 |
| scaleedit / `scaleedit-f33862e63a00987c4461816e` | Change the color of the green icing bag and the icing on the left cupcake to teal. | Ambiguous existing-mask binding for unit_1_1 |
| scaleedit / `scaleedit-51e58be9c7a4878a9960cb3b` | Change the color of the woman's blue headscarf and waist cloth to purple. | An operation has no unambiguously matched existing mask |
| scaleedit / `scaleedit-75087daa097d6e3642a38e48` | Remove two monkeys from the image, leaving only the monkey sitting on top of the central black pole. | An operation has no unambiguously matched existing mask |
| scaleedit / `scaleedit-7c64bc4d539da809af2cb6c7` | Add one red block to the top of the red stack. | Need exactly one this region for each reference, in original order |
| scaleedit / `scaleedit-f5e02ac3ba788a7a609c4cd3` | Transform the material of the two yellow buildings on the left and right into polished stainless steel. | Reference must match exactly once: 'two yellow buildings on the right' (0 matches) |
| scaleedit / `scaleedit-0659751fe06079168db67b27` | Repair the broken skateboard on the right and remove the graphic stickers to reveal the bare wood. | Need exactly one this region for each reference, in original order |
| scaleedit / `scaleedit-d0d09227f340edae1746bb2c` | Draw shading in the darkest overlapping central region where sets A, B, C, and D intersect. | Add reference must include NEW content and placement, not only the existing carrier |
| scaleedit / `scaleedit-c58380db18910ebd555d06cf` | Remove the two black dots at the intersection points of the line and circle, and add one black dot at the center of the circle on the line. | Ambiguous existing-mask binding for unit_0_0 |
| scaleedit / `scaleedit-750101894c07040b9ac40efd` | Add more fully open and larger petals around the sunflower in the center to give it the appearance of being in full bloom. | Add reference must include NEW content and placement, not only the existing carrier |
| scaleedit / `scaleedit-93943e1a3a5823dc45e1e7b7` | "Reduce the size of the large bag on the left to match the size of the medium bag on the right." | Need exactly one this region for each reference, in original order |
| scaleedit / `scaleedit-6b9dec76b9da12c97e47600a` | Make the Coca-Cola bottle on the left smaller than the one on the right. | Need exactly one this region for each reference, in original order |
| scaleedit / `scaleedit-91871e94d2cfb3e0a59f6c02` | Make the Coca-Cola bottle in the background smaller than the one in the foreground. | Need exactly one this region for each reference, in original order |
| scaleedit / `scaleedit-871d18f0dfd48069a8ce5fe1` | Make the smaller card on the right the same dimensions as the larger card on the left. | An operation has no unambiguously matched existing mask |
| scaleedit / `scaleedit-389baba8491674e8eaecbc26` | Add a small wooden cross, an open Bible, and a lit candle on the table near the flowers. | Need exactly one this region for each reference, in original order |
| scaleedit / `scaleedit-63b5a7a1e6f131ccacb7aaf3` | Draw a right-facing arrow in the '?' box on the right. | Add reference must include NEW content and placement, not only the existing carrier |
| scaleedit / `scaleedit-2aecd98d1aa8141fa27c27f7` | Replace the '?' in the rightmost figure with the number '40'. | Text rewrite retained OLD quoted text |
| scaleedit / `scaleedit-9b669b02915a5fba18b47946` | Add a red 'O' in the bottom center spot on the tic-tac-toe board. | Add reference must include NEW content and placement, not only the existing carrier |
| scaleedit / `scaleedit-4ac9adcd053232c72012a8cb` | Change the reading on the foreground glucose meter display to "3.0". | Reference must match exactly once: "'3.0'" (0 matches) |
| scaleedit / `scaleedit-b5c41123d12aeeead12512d1` | "Add visible patches of green cyanobacteria blooms to the surface of the water near the shoreline and around the island." | Need exactly one this region for each reference, in original order |
| scaleedit / `scaleedit-7af2a702c506a616d4a2eca3` | Make the person near the left side of the image appear to be walking briskly. | Add reference must include NEW content and placement, not only the existing carrier |
| crispedit / `crispedit-b65dfb1fd1b36979999164a4` | remove the whole lemon and the sliced lemon from the plate | Ambiguous existing-mask binding for source_0 |
| crispedit / `crispedit-1a8013f8ce222f05b92eef25` | Turn individuals positioned in the right area into darker skin tone and update vests to orange | Ambiguous existing-mask binding for source_0 |
| crispedit / `crispedit-5594a84a285f382a7ae54843` | remove Woody, Buzz Lightyear, and Jessie from the birthday party setup | Ambiguous existing-mask binding for source_2 |
| scaleedit / `scaleedit-0e1b27de2b4dfc3267fc3ba0` | Remove the man in the green coat, and change the color of the car on the right to red. | Ambiguous existing-mask binding for unit_0_0 |
| scaleedit / `scaleedit-64bb962ae7287f9ef2a4a952` | Remove the person sitting on the grass, and change the color of the standing person's hoodie to light blue. | Ambiguous existing-mask binding for unit_0_0 |
| scaleedit / `scaleedit-229b01272f5beb8b94663379` | Add a drop of blue ink to the surface of the tea in the left cup and mix blue ink into the tea in the right cup. | Ambiguous existing-mask binding for unit_1_1 |
| scaleedit / `scaleedit-739cf0e2f2d0849974feb8fd` | Remove the gondola on the left side of the image, and change the color of the boat in the center to red. | Ambiguous existing-mask binding for unit_2_2 |
| scaleedit / `scaleedit-4ee43583505b552dcd6942ec` | Remove the small boat near the center-left of the image, and change the color of the large ship on the right to red. | An operation has no unambiguously matched existing mask |
| scaleedit / `scaleedit-b01ad5f10ecf8e237d6b38d9` | Invert the black and white triangles in the center diamond of the grid. | Cannot refine coarse dataset type from the operation |
| scaleedit / `scaleedit-af454ae0eae96a583106a8b6` | Fill the central rectangular cross-section of the polyhedron with red on the left and yellow on the right. | Cannot refine coarse dataset type from the operation |
| crispedit / `crispedit-3ffe8ca54d42ac1518821de2` | The subjects go from high-fiving to clasping their hands together with a surprised expression. | Missing original words: go. Keep old locating details in ref_phrase and new content/constraints in noref_instruction |
| scaleedit / `scaleedit-839a94ce9a5b724dd28b9ee1` | Change the color of the root beer stand tent and the lemonade stand tent to blue and white stripes. | Ambiguous existing-mask binding for unit_1_1 |
| crispedit / `crispedit-dd37d2e438a07ecda80b5632` | remove the basket of clothes and the fur coat on the floor | Reference must match exactly once: 'basket of clothes on the floor' (0 matches) |
| scaleedit / `scaleedit-c7503b12f35eaa23907c72fd` | Transform the wings of the red moth labeled A into glass. | Protocol conversion rejected annotation: {'ref': 'Mask group cannot directly follow an article/preposition'} |
| crispedit / `crispedit-d2b469f88e63f1eac1d4a6cb` | replace the blue Model A with a red sports car | Protocol conversion rejected annotation: {'ref': 'Mask group cannot directly follow an article/preposition'} |
| scaleedit / `scaleedit-679032cb28043b7c6f5eace4` | Change the letters on the top two blocks from 'A' and 'B' to 'X' and 'Y'. | Text unit must identify one original quoted string |
| scaleedit / `scaleedit-0bff0b8a51ae1a301451cf5a` | Restore the faces of both children and change the shirt color of the child on the right to red. | The region already denotes the selected part; remove the repeated part-of operator |
| crispedit / `crispedit-b08fa4e35b39776ee407c8b4` | A character lowers their pointing arm. | Missing original words: their. Keep old locating details in ref_phrase and new content/constraints in noref_instruction |
| scaleedit / `scaleedit-dce91b75213a1b3ed3692bbf` | Add a second cup of coffee and a second spoon to the black mat next to the existing ones. | Add reference must include NEW content and placement, not only the existing carrier |
| scaleedit / `scaleedit-8c7487d6fc705fab2a29c8f5` | Change the height of the smallest and largest stools to match the height of the medium stool. | Reference must match exactly once: 'smallest stool' (0 matches) |
| scaleedit / `scaleedit-58776341a263e97139edd7c8` | Draw a frowning face (two eyes and a downward curved mouth) in the bottom right square. | Add reference must include NEW content and placement, not only the existing carrier |

## 3. 程序接受、但我审阅不通过的 9 条

这些记录会出现在自动 accepted 候选中，是当前最需要关注的漏检。下面每行给出原文、模型实际两字段及具体原因。

### 3.1 `crispedit-44e3b0a6fc67a44a17133c07`

原文：`add Liam Hemsworth standing next to Miley Cyrus`

`ref_phrase`: `["Liam Hemsworth standing next to Miley Cyrus"]`

`noref_instruction`: `Add Liam Hemsworth standing next to Miley Cyrus in this region.`

审阅原因：Original next-to-Miley placement retained in noref。

### 3.2 `scaleedit-08023bf3bd5eccc67d37a76c`

原文：`Remove the middle lunch box from the stack so that only the two largest boxes remain.`

`ref_phrase`: `["middle lunch box"]`

`noref_instruction`: `Remove this region from the stack so that only the two largest boxes remain.`

审阅原因：Old from-the-stack locator remains in noref。

### 3.3 `scaleedit-49285969e0cb0a17d8424b52`

原文：`Add a large, colorful hot air balloon floating in the sky above the castle, slightly to the left of the center.`

`ref_phrase`: `["large, colorful hot air balloon floating in the sky above the castle, slightly to the left of the center"]`

`noref_instruction`: `Add a large, colorful hot air balloon floating in the sky above the castle, slightly to the left of the center in this region.`

审阅原因：Original sky/castle/left-of-center placement remains in noref。

### 3.4 `scaleedit-f7fa8ca35e994e400a0a3da8`

原文：`Add a black and white dog sitting on the sidewalk to the left of the white sign.`

`ref_phrase`: `["black and white dog sitting on the sidewalk to the left of the white sign"]`

`noref_instruction`: `Add a black and white dog sitting on the sidewalk in this region.`

审阅原因：Original sidewalk placement remains in noref。

### 3.5 `scaleedit-403d6f2b64b43fa6fa9f3c28`

原文：`Add a brown smudge to the top-center leaf and a black speckled spot to the bottom-center leaf of the plant.`

`ref_phrase`: `["top-center leaf", "bottom-center leaf"]`

`noref_instruction`: `Add a brown smudge to this region and a black speckled spot to this region of the plant.`

审阅原因：Add references select existing leaves instead of NEW contents and placements; old plant qualifier remains。

### 3.6 `crispedit-309578b6ef7429ebbf2d7f4e`

原文：`Add a unique side table near the lower right side of the image between the sofa and loveseat. Location and size: Lower right quadrant, approximately 209x210 pixels.`

`ref_phrase`: `["unique side table near the lower right side of the image between the sofa and loveseat"]`

`noref_instruction`: `Add a unique side table in this region. Location and size: Lower right quadrant, approximately 209x210 pixels.`

审阅原因：Old lower-right quadrant metadata remains in noref。

### 3.7 `scaleedit-1fc40823ed1daf807370dc64`

原文：`Remove the two outer red dots from the horizontal line, leaving only the center one.`

`ref_phrase`: `["two outer red dots from the horizontal line"]`

`noref_instruction`: `Remove this region from the horizontal line, leaving only the center one.`

审阅原因：Old horizontal-line locator remains in noref。

### 3.8 `scaleedit-68f1eed7ca2846d602e2516f`

原文：`Replace the text 'DEC 10 MON 8/7c' with 'NOV15-TUE9/8P' under the ARROW title.`

`ref_phrase`: `["'DEC 10 MON 8/7c'"]`

`noref_instruction`: `Replace this region with 'NOV15-TUE9/8P' under the ARROW title.`

审阅原因：Old under-ARROW carrier locator remains in noref。

### 3.9 `scaleedit-01eb5459e758ff446834c739`

原文：`Raise the red liquid level in the second thermometer from the left.`

`ref_phrase`: `["the second thermometer from the left"]`

`noref_instruction`: `Raise the red liquid level in this region.`

审阅原因：Reference selects thermometer container rather than edited liquid level。

## 4. 我审阅不通过、同时被程序拒绝的 9 条

这 9 条已经被程序拦住，不能计入 accepted。下面列出模型输出以说明实际错误形式。

- `crispedit-7e5d9977e38dc89b0db0373d`：The person lifts the kitten closer to their face.；模型输出 `ref=["person", "kitten"]`、`noref=This region lifts this region closer to their face.`。原因：Action participant kitten becomes a separate edited region。
- `scaleedit-75087daa097d6e3642a38e48`：Remove two monkeys from the image, leaving only the monkey sitting on top of the central black pole.；模型输出 `ref=["two monkeys", "the monkey sitting on top of the central black pole"]`、`noref=Remove this region, leaving only this region.`。原因：Unchanged surviving monkey becomes an extra edited region。
- `scaleedit-93943e1a3a5823dc45e1e7b7`："Reduce the size of the large bag on the left to match the size of the medium bag on the right."；模型输出 `ref=["large bag on the left", "medium bag on the right"]`、`noref=Reduce the size of this region to match the size of the medium bag on the right.`。原因：Unchanged comparison bag receives an extra reference。
- `scaleedit-6b9dec76b9da12c97e47600a`：Make the Coca-Cola bottle on the left smaller than the one on the right.；模型输出 `ref=["Coca-Cola bottle on the left", "one on the right"]`、`noref=Make this region smaller than the one on the right.`。原因：Unchanged comparison bottle receives an extra reference。
- `scaleedit-389baba8491674e8eaecbc26`：Add a small wooden cross, an open Bible, and a lit candle on the table near the flowers.；模型输出 `ref=["a small wooden cross", "an open Bible", "a lit candle", "on the table near the flowers"]`、`noref=Add a small wooden cross, an open Bible, and a lit candle in this region.`。原因：Four references but only one region; placement split into extra unit。
- `scaleedit-63b5a7a1e6f131ccacb7aaf3`：Draw a right-facing arrow in the '?' box on the right.；模型输出 `ref=["'?' box on the right"]`、`noref=Draw a right-facing arrow in this region.`。原因：Add reference selects existing box instead of NEW arrow with placement。
- `scaleedit-dce91b75213a1b3ed3692bbf`：Add a second cup of coffee and a second spoon to the black mat next to the existing ones.；模型输出 `ref=["black mat"]`、`noref=Add a second cup of coffee and a second spoon to this region next to the existing ones.`。原因：Add reference selects existing mat rather than NEW cup/spoon and placement。
- `scaleedit-8c7487d6fc705fab2a29c8f5`：Change the height of the smallest and largest stools to match the height of the medium stool.；模型输出 `ref=["smallest stool", "largest stool", "medium stool"]`、`noref=Change the height of this region and this region to match the height of this region.`。原因：Reference invents singular phrases absent from source and selects unchanged comparison stool。
- `scaleedit-58776341a263e97139edd7c8`：Draw a frowning face (two eyes and a downward curved mouth) in the bottom right square.；模型输出 `ref=["bottom right square"]`、`noref=Draw a frowning face (two eyes and a downward curved mouth) in this region.`。原因：Add reference selects existing square rather than NEW face and placement。

## 5. 文本审阅可接受、程序拒绝的 2 条

这两条显示确定性检查也有误拒。不会自动放行，因为还需要核对相应类型与实例绑定；本轮保留在失败队列。

- `scaleedit-5505b1c66abb2645c581e3a1`：Spread the wings of the angel statue on the left side of the bridge and extend its right arm outward.；输出 `ref=["wings of the angel statue on the left side of the bridge", "right arm"]`、`noref=Spread this region and extend this region outward.`。程序最终错误：An operation has no unambiguously matched existing mask。
- `crispedit-b08fa4e35b39776ee407c8b4`：A character lowers their pointing arm.；输出 `ref=["pointing arm"]`、`noref=A character lowers this region.`。程序最终错误：Missing original words: their. Keep old locating details in ref_phrase and new content/constraints in noref_instruction。

审阅明细、全 122 条原始输出和每个 accepted/failed 状态另保存在 `/tmp/samtok21-qwen35-compare/reviewed_outputs.jsonl`。分母、挑样方法、程序结果与性能测试见[对照实验](SAMTokEdit_Qwen21_noref提示词优化与模型复测.md#4-同一新环境下三个模型的实测)。
