# SAMTok Qwen-Image-2.1 独立案例研究（不参考 Judge 分数）

本报告只根据 source、target（有提供时）、Qwen-Image-2.1 baseline 输出、SAMTok Stage 2 step-12000 输出和 final 输出的图像内容做独立判断。写作和筛选过程不读取 Judge 的 E/P/Q/strict 分数；Judge 结果仅作为已有实验记录中的另一条自动化证据，不参与本报告的标签。

本次先对 656 个 case 的 mask setting 生成了全量缩略图巡检，再按数据集和编辑类型均衡抽取 48 个 case，逐一查看四种 setting（text-only、mask、box、point）下的 source / target / SAMTok token decode / baseline / step-12000 / final。详细图像位于评测结果目录的 `independent_case_study/figures/`。MIRAGE 没有统一 target 图时，按 instruction 和 source 的局部结构判断是否完成及是否过度修改。

## 1. 覆盖范围与案例清单

### 1.1 全量数据分布

| 数据集 | add | remove | replace | mixed | 合计 |
|---|---:|---:|---:|---:|---:|
| CompBench | 255 | 255 | 22 | 0 | 532 |
| HumanEdit | 3 | 10 | 11 | 0 | 24 |
| MIRAGE | 2 | 3 | 61 | 34 | 100 |
| 合计 | 260 | 268 | 94 | 34 | 656 |

### 1.2 全量缩略图巡检总览

下图是按编辑类型整理的 mask setting 全量缩略图巡检结果。每个小格从左到右为 source、baseline、SAMTok final；它用于发现系统性模式，具体结论以随后 48 个四 setting 案例图为准。

![全量 add 巡检](<assets/independent_case_study/overview/add_mask_annotation_00.jpg>)

![全量 remove 巡检](<assets/independent_case_study/overview/remove_mask_annotation_00.jpg>)

![全量 replace 巡检](<assets/independent_case_study/overview/replace_mask_annotation_00.jpg>)

![全量 mixed/后续巡检](<assets/independent_case_study/overview/mixed_mask_annotation_00.jpg>)

注：上述临时总览图用于本次人工巡检，最终可复现的重点案例图已复制到评测结果目录；文档中的详细案例图均使用结果目录内的持久化文件。

### 1.3 逐案例清单与独立判断

下表列出本次详细查看的全部 48 个案例。`优`表示 SAMTok 在空间绑定、编辑完成度或区域外保持上明显优于 baseline；`劣`表示明显失败或 baseline 更可靠；`近`表示两者都完成或差异主要是细节。该标签是人工视觉判断，不是 Judge 分数。

| 编号 | 数据集/类型 | 指令摘要 | 独立结论 |
|---:|---|---|---|
| 0000 | compbench / add | add a gray fish on the leftmost with its head downward | **劣**：四种 setting 都能保持水箱和右侧文字，但新增鱼在 text/mask/box/point 中均不稳定；final 多数没有形成目标的倒头灰鱼。baseline 至少在 text/box 中生成了鱼，说明这是新增实体和姿态约束的明显失败例。 |
| 0036 | compbench / add | add a similar zebra standing in the opposite direction on the left fro | **优**：显式 mask/box/point 能在第二只斑马前面形成新斑马，方向大体接近目标；text-only 更容易只保留原群。SAMTok 的位置绑定优于 baseline 的大范围改动，但新实例姿态仍不稳定。 |
| 0073 | compbench / add | add a bird similar to the others on the topmost | **劣**：显式区域下 final 会在上方生成鸟，空间位置基本对；部分 setting 只生成小鸟或数量不足。baseline 容易一次生成多只鸟，SAMTok 更克制但存在 under-edit。 |
| 0109 | compbench / add | add a black duck similar in size and orientation to the second duck fr | **近**：黑鸭新增在目标鸭上方，显式 setting 的 final 位置和尺度较接近目标；baseline 也能生成，但有时覆盖已有鸭。SAMTok 的区域约束更稳，细节质量相当。 |
| 0145 | compbench / add | add a plane on the rightmost side that is similar to the other planes | **优**：这是新增飞机的较好例子。final 在 text/box/point 中增加了右侧白色飞机，且没有改变原有战斗机；baseline 的 mask 输出仍残留红色 locator 标记。 |
| 0181 | compbench / add | add a grey cat on the right side of the cat with its back to the viewe | **劣**：目标要求在右侧新增一只背对观众的灰猫。SAMTok 多数 setting 只出现模糊残影或错误大小的猫，baseline 的显式 setting 更接近目标；新增主体与原猫相邻时仍会发生绑定/尺度失败。 |
| 0218 | compbench / add | add a bird-like object on the rightmost side with its head pointing to | **近**：目标右侧应出现头朝右下的鸟状物。SAMTok 显式 setting 常保持原两只鸽子而不新增，或只留下灰雾；baseline 在 mask/box 中更接近新增主体。 |
| 0233 | compbench / remove | remove the leftmost fish | **近**：删除左侧鱼时，SAMTok 的四种 setting 都能留下右侧橙鱼和水箱背景，baseline 在部分显式输入下仍保留或重绘左鱼。这个例子体现了最稳定的 remove 优势。 |
| 0269 | compbench / remove | remove the chicken at the lower leftmost | **近**：显式 mask/box/point 能只删除左下目标鸡并保留其余鸡；text-only 会把鸡群重排甚至引入白鸡。SAMTok 需要显式区域才能稳定利用空间绑定。 |
| 0306 | compbench / remove | remove the first yellow fish from below | **近**：删除下方第一条黄鱼时，SAMTok 显式输出能清理目标区域并保持珊瑚和红鱼；baseline 更容易留下目标或出现局部残影。 |
| 0342 | compbench / remove | remove the rightmost sheep | **近**：删除最右羊的显式 final 基本保持羊群和雪地，局部边界略软；baseline 仍有目标羊残留。SAMTok 的区域外保真较好。 |
| 0378 | compbench / remove | remove the leftmost duck in the group of ducks | **优**：显式输入下 final 能删除左侧鸭并保留右侧鸭群；text-only 会扩大擦除范围。baseline 的定位也不稳定，SAMTok 的显式 token 更可靠。 |
| 0414 | compbench / remove | remove the leftmost cow | **优**：显式 final 能清理左侧牛所在区域，同时保留道路、雪和行人；baseline 在部分 setting 仍残留牛。属于 remove 的清晰正例。 |
| 0451 | compbench / remove | remove the leftmost panda | **近**：删除左下熊猫时，SAMTok 显式结果保留上方熊猫和篮筐，背景补全相对自然；text-only 会把树枝区域擦得更宽。 |
| 0470 | compbench / replace | replace the person on the right with a person in white | **近**：替换右侧骑行者为白衣人时，SAMTok 四种 setting 都能把目标人物改为白衣并保持道路结构；baseline 也能完成，但 SAMTok 的显式边界更干净。 |
| 0475 | compbench / replace | replace the third giraffe from the right with background | **近**：把第三只长颈鹿替换为背景时，SAMTok 显式 final 的目标区域基本干净，step 中的白色鬼影在 final 减少；baseline 显式结果更容易留下轮廓或 marker。 |
| 0480 | compbench / replace | replace the leftmost rabbit with a small table model | **近**：兔子替换为小桌子时，SAMTok 显式结果能放置桌子，位置正确但桌型/颜色与 target 不同；baseline 往往桌子过大或同时重绘旁边兔子。 |
| 0486 | compbench / replace | replace the rightmost car with a black car moving to the right | **近**：右侧车辆改为黑车是稳定 replace。SAMTok 显式 final 的车辆位置、道路和树木保持较好，baseline 也能完成，差异主要在车辆细节。 |
| 0491 | compbench / replace | replace the red motorcycle on the right with a black motorcycle | **近**：右侧摩托改黑色时，SAMTok 的显式 final 接近 target，未明显改变左侧摩托；baseline 也能完成但有时形状更大。 |
| 0513 | compbench / add | add a chicken in the middle and a chicken on the bottom left | **近**：双实例新增是相对成功的 add：显式 setting 能在中部和左下放置两只鸡，目标数量大体正确；baseline 经常生成过大的鸡或保留 locator。SAMTok 位置更受控，但颜色、姿态和边缘仍有偏差。 |
| 0531 | compbench / remove | remove the chicken in the middle and the chicken on the bottom left | **近**：双目标删除的显式 final 能去掉中部和左下鸡，但 text-only/step 会出现大块模糊或新白鸡，且部分 final 对剩余鸡的数量不够稳定。多目标 remove 比单目标更容易过度编辑。 |
| 0532 | humanedit / replace | Replace the black hat of the big doll in the middle with a Santa hat | **近**：Santa 帽语义完成，但 SAMTok final 的帽子偏大并压住人偶脸部；baseline 的帽形相对接近 target。局部属性替换需要更细的边界和尺度控制。 |
| 0533 | humanedit / add | Add a white cat next to the gray cat. | **近**：显式 mask/box/point 能在灰猫旁新增白猫，位置和数量较好；baseline 可完成但容易保留红框。SAMTok 在这个 add 小物体案例中表现出可用的空间控制。 |
| 0535 | humanedit / add | Add a white strawberry to the middle of several red strawberries. | **劣**：白草莓 add 是明显失败：SAMTok final 多数只生成白色团块或改变原草莓，没有形成清晰的白草莓；baseline 更接近目标形状但仍有红框。 |
| 0536 | humanedit / remove | Remove the little boy next to the little girl. | **近**：删除小男孩时，显式 final 能保留小女孩和背景，区域外结构自然；baseline 部分 setting 未完全删除。 |
| 0539 | humanedit / replace | Replace the snowflake pattern on the second block from the left with t | **近**：把第二块雪花改为 Y 时，SAMTok 显式 final 能准确改变字母且保持其他方块；baseline 也能完成。细粒度文字替换在本例可用。 |
| 0544 | humanedit / add | Add a bee in the center of the flower in the middle of the picture | **劣**：在花中心新增蜜蜂时，SAMTok 多数 setting 不生成蜜蜂或只留下很小模糊斑，baseline 至少生成了蜜蜂但尺寸偏大。新增细小主体是当前弱项。 |
| 0545 | humanedit / remove | Remove all the people behind the man in the center of the picture | **近**：删除中心人物后方的人时，SAMTok 显式 final 能保留前景人物并修复背景；baseline 更容易保留后方人。 |
| 0546 | humanedit / replace | Replace the hat worn by the person on the far left with a white hat. | **近**：把最左人物帽子改白时，SAMTok final 的显式输出准确修改目标帽子，其他人物基本保持；baseline 也能完成。 |
| 0551 | humanedit / replace | Replace the empty plate on the left side of the table with a plate wit | **近**：把左侧空盘换成有食物的盘子时，SAMTok 显式 final 放置了食物盘，位置对但玻璃杯和桌面有轻微联动变化；baseline 改动更宽。 |
| 0552 | humanedit / remove | Remove the tallest tree in the middle of the forest. | **优**：删除森林中央最高树是非常清晰的成功例，SAMTok final 的显式输出与 target 接近，baseline 多数仍保留树。 |
| 0555 | humanedit / remove | Remove all but the fallen volleyball player | **优**：要求只保留倒地排球运动员。SAMTok 显式结果比 baseline 更接近，但仍常保留右侧站立人物或留下擦除痕迹；多对象保留/删除边界仍不完整。 |
| 0556 | mirage / mixed | Change the color of the shell of the leftmost turtle to red, and add s | **优**：显式 token 能把左龟变红、右龟加雪，目标对象绑定比 baseline（多只龟变红）更准确；雪会溢出到路面，说明区域外保真仍有限。 |
| 0557 | mirage / replace | Change the color of the middle car's headlights to red, and change the | **劣**：车辆颜色属性的 final 能部分改变中间车灯，但右车 logo 的蓝色不稳定，属于 under-edit；baseline 更激进，容易把多辆车一起改色。 |
| 0579 | mirage / mixed | Add a bunch of red roses to the middle person's hand, and change the c | **劣**：final 能放置红玫瑰并部分改变右侧裤子，但裤色/位置不总是完整；baseline 也常把多个区域一起染色。SAMTok 的多 span 空间控制较好但属性完成不足。 |
| 0581 | mirage / remove | Remove the middle bird's wings. | **劣**：删除中间鸟的翅膀时，SAMTok 显式 final 经常直接删除整只鸟，而不是保留无翅身体；这是局部结构编辑失败，baseline 同样不稳定。 |
| 0585 | mirage / replace | Change the color of the left dog's eyes to blue, and change the right  | **劣**：左狗眼睛变蓝、右狗毛变粗时，SAMTok 显式 final 常把多只狗整体染蓝，目标绑定失败；baseline 也会扩大编辑，但相对更接近局部属性。 |
| 0590 | mirage / mixed | Change the color of the middle person's shirt to yellow, and add some  | **优**：中间衣服变黄、左衣服加熔岩时，SAMTok 显式 final 能分别命中两个主体，整体比 baseline 的多人物过度编辑更克制；熔岩边界和衣服纹理仍偏粗。 |
| 0605 | mirage / mixed | Change the color of the middle person's shirt to green, and add some s | **近**：中间衣服改绿的空间绑定较好，左肩雪有时很弱或缺失；baseline 容易把多个衣服一起改绿。 |
| 0608 | mirage / add | Add a blue scarf to the first cat from the left, and add a red scarf t | **近**：给两只猫加不同颜色围巾时，SAMTok final 多数不生成清晰围巾，baseline 显式结果反而能呈现蓝/红围巾。多主体、多颜色 add 明显困难。 |
| 0611 | mirage / add | Add some snow onto the head of the second horse from the right, and ad | **劣**：雪和熔岩都能被生成，但 SAMTok 常把效果放到相邻马或扩大到头顶之外；baseline 更容易全体加火焰。SAMTok 更克制但目标身份仍不稳定。 |
| 0615 | mirage / mixed | Add a scarf to the neck of the first dog from the right, and remove th | **劣**：同时新增右侧狗围巾并删除另一只围巾时，SAMTok final 往往两项都不完整，保持原图；复杂 add+remove 组合需要更强的 clause 绑定。 |
| 0622 | mirage / mixed | Add a rose pattern onto the chest of the first sheep from the left, an | **优**：显式 final 能在左羊胸口形成玫瑰、右羊耳朵变粉，且比 baseline 给所有羊加花/粉耳更局部；这是 mixed 属性编辑的较好例子。 |
| 0627 | mirage / replace | Change the color of the shirt of the first person from the left to blu | **优**：第一人衣服变蓝的目标通常能完成，第二人裤子材质变化较难从视觉上稳定呈现；SAMTok 比 baseline 更少改动其他人，但材质编辑仍缺乏可见一致性。 |
| 0628 | mirage / mixed | Change the screen display of the second machine from the right to a be | **近**：机器屏幕改海滩、另一台加熔岩的复杂 mixed 任务中，SAMTok final 常只生成大块熔岩，屏幕没有按要求变化；baseline 也有大范围火焰但部分屏幕变化更明显。 |
| 0651 | mirage / remove | Remove the first snowmobile from the left, and remove the dirt on the  | **劣**：删除左侧雪地车和第三车前污渍时，SAMTok 显式 final 多数删掉目标车并保留右侧车辆；部分 setting 会把道路/水雾一起抹掉，污渍清理不稳定。 |
| 0654 | mirage / replace | Change the material of the lanyard of the second person from the left  | **优**：第五人的头发变白、第二人的挂绳材质变化在显式 final 中较可见，其他人基本保持；属于细粒度属性编辑的相对成功例。 |
| 0655 | mirage / mixed | Change the color of the eyes of the first sparrow from the left to gre | **劣**：要求第一只麻雀眼睛变绿并删除第三只白羽时，SAMTok 往往把多只鸟变绿或直接抹掉鸟，局部属性和局部删除同时出现时失败明显。 |

## 2. 逐案例图像与观察

## 2.1 mask token 的来源与图中的 token decode 列

图中新增的 **SAMTok token decode** 列不是生成结果，也不是 benchmark 的 target。它是把实际送入 DiT 的 prompt 中的 `<|mt_start|>...<|mt_end|>` token span 再交给冻结的 SAMTok codec 解码，并将解码出的区域以红色叠加到 source 上。因而它回答的是“模型实际收到的空间区域在哪里”，可以用来检查 token 编码/插入是否正确。生成脚本在评测结果目录的 `code/build_case_package.py:134-144`，核心调用是 `codec.decode(source, token_prompt)`；训练/推理条件构造在评测结果目录的 `code/samtok_stage2_benchmark_noref_aligned.py:242-302`。

不同 setting 的 token 来源如下：

- **text-only**：先把原始带位置描述的 `with_location_reference` 指令交给 Stage 1 ref-localization（`samtok_stage2_benchmark_noref_aligned.py:266-272`）。定位器成功时，会把预测的区域 span 插回原始语言 prompt，图中的 token decode 展示这个预测区域；如果定位器没有产生 span，则记录为 `mode=direct`，prompt 可能没有可解码的 mask token。
- **mask_annotation**：直接读取 benchmark 已提供的 region mask（`...:279-283`），用 SAMTok codec 编码成 token span。这里的输入 mask 就是 benchmark 的数据 mask，不再重新推断或校正。
- **box_annotation**：把 benchmark 的 box 交给 SAMTok/SAM2 segmentation，取 score 最高的 proposal（`...:284-287`），再用 codec 编码。
- **point_annotation**：把 benchmark 的 point 交给 segmentation，取 score 最高的 proposal（`...:288-292`），再用 codec 编码。
- **多区域 case**：每个 region 单独得到一个 span，并按原始 region 顺序重排（`...:293-297`）；noref 编译器再将各 span 插入对应的 atomic clause。token decode 列会把所有 span 解码出的区域叠加显示。

因此，token decode 只验证“token 所表达的空间区域”，不能直接证明最终编辑完成；mask/box/point 的 token decode 也可能因为其输入方式不同而不完全相同。

## 2.2 text-only 与显式区域 setting 的 DiT prompt

两类 setting 共享同一套 mask-token 边界语法和同一个 DiT 编辑接口：source image 加 text prompt，prompt 中有区域时使用 `<|mt_start|>...<|mt_end|>`。但完整 prompt 字符串、token 的来源和 conditioning route 并不相同。

- **text-only** 使用 `prompt_variant=text_only_ref_localization`：保留原始 `with_location_reference` 的自然语言描述，由 Stage 1 定位器在原句中插入预测 span；成功时是 `mode=inline`，失败 fallback 时是 `mode=direct`，可能完全不含 mask token。
- **mask/box/point** 使用 `prompt_variant=samtok_native_noref_aligned_v1`：由 `build_aligned_noref_prompt`（`...:171-239`）生成 noref 模板，例如 `Remove <token span>.`、`Add ... in this region <token span>.`、`Change this region <token span> to red.`，统一走 `mode=inline`。mask/box/point 的差别只在 token span 的空间来源。

例如 case `0000` 的 text-only prompt 保留“add a gray fish on the leftmost ...”的原始位置语言，显式 mask prompt 则是 `Add a gray fish ... in this region <span>.`；两者的 token 边界格式一致，但自然语言模板和区域来源不同。所以不能把 text-only 与其他 setting 说成“prompt 完全相同”，更准确的说法是：它们在 DiT 侧兼容同一 token 接口，但分别测试语言定位和显式区域条件。

## 2.3 详细图的列定义

每张图包含四行 setting；每行六列依次为 **source、target、SAMTok token decode、baseline、step-12000、final**。其中 token decode 是输入区域可视化，baseline/step-12000/final 才是三组生成输出。图中 target 为空表示 benchmark 没有统一 target 文件（主要是 MIRAGE），此时按照 instruction 判断。


### 0000｜compbench｜add

**指令：** add a gray fish on the leftmost with its head downward

![案例 0000 对比](<assets/independent_case_study/figures/0000_add_compbench.jpg>)

**独立观察：** 四种 setting 都能保持水箱和右侧文字，但新增鱼在 text/mask/box/point 中均不稳定；final 多数没有形成目标的倒头灰鱼。baseline 至少在 text/box 中生成了鱼，说明这是新增实体和姿态约束的明显失败例。

### 0036｜compbench｜add

**指令：** add a similar zebra standing in the opposite direction on the left front of the second zebra from the rightmost

![案例 0036 对比](<assets/independent_case_study/figures/0036_add_compbench.jpg>)

**独立观察：** 显式 mask/box/point 能在第二只斑马前面形成新斑马，方向大体接近目标；text-only 更容易只保留原群。SAMTok 的位置绑定优于 baseline 的大范围改动，但新实例姿态仍不稳定。

### 0073｜compbench｜add

**指令：** add a bird similar to the others on the topmost

![案例 0073 对比](<assets/independent_case_study/figures/0073_add_compbench.jpg>)

**独立观察：** 显式区域下 final 会在上方生成鸟，空间位置基本对；部分 setting 只生成小鸟或数量不足。baseline 容易一次生成多只鸟，SAMTok 更克制但存在 under-edit。

### 0109｜compbench｜add

**指令：** add a black duck similar in size and orientation to the second duck from the right on its upper side

![案例 0109 对比](<assets/independent_case_study/figures/0109_add_compbench.jpg>)

**独立观察：** 黑鸭新增在目标鸭上方，显式 setting 的 final 位置和尺度较接近目标；baseline 也能生成，但有时覆盖已有鸭。SAMTok 的区域约束更稳，细节质量相当。

### 0145｜compbench｜add

**指令：** add a plane on the rightmost side that is similar to the other planes

![案例 0145 对比](<assets/independent_case_study/figures/0145_add_compbench.jpg>)

**独立观察：** 这是新增飞机的较好例子。final 在 text/box/point 中增加了右侧白色飞机，且没有改变原有战斗机；baseline 的 mask 输出仍残留红色 locator 标记。

### 0181｜compbench｜add

**指令：** add a grey cat on the right side of the cat with its back to the viewer and facing left

![案例 0181 对比](<assets/independent_case_study/figures/0181_add_compbench.jpg>)

**独立观察：** 目标要求在右侧新增一只背对观众的灰猫。SAMTok 多数 setting 只出现模糊残影或错误大小的猫，baseline 的显式 setting 更接近目标；新增主体与原猫相邻时仍会发生绑定/尺度失败。

### 0218｜compbench｜add

**指令：** add a bird-like object on the rightmost side with its head pointing to the lower right corner

![案例 0218 对比](<assets/independent_case_study/figures/0218_add_compbench.jpg>)

**独立观察：** 目标右侧应出现头朝右下的鸟状物。SAMTok 显式 setting 常保持原两只鸽子而不新增，或只留下灰雾；baseline 在 mask/box 中更接近新增主体。

### 0233｜compbench｜remove

**指令：** remove the leftmost fish

![案例 0233 对比](<assets/independent_case_study/figures/0233_remove_compbench.jpg>)

**独立观察：** 删除左侧鱼时，SAMTok 的四种 setting 都能留下右侧橙鱼和水箱背景，baseline 在部分显式输入下仍保留或重绘左鱼。这个例子体现了最稳定的 remove 优势。

### 0269｜compbench｜remove

**指令：** remove the chicken at the lower leftmost

![案例 0269 对比](<assets/independent_case_study/figures/0269_remove_compbench.jpg>)

**独立观察：** 显式 mask/box/point 能只删除左下目标鸡并保留其余鸡；text-only 会把鸡群重排甚至引入白鸡。SAMTok 需要显式区域才能稳定利用空间绑定。

### 0306｜compbench｜remove

**指令：** remove the first yellow fish from below

![案例 0306 对比](<assets/independent_case_study/figures/0306_remove_compbench.jpg>)

**独立观察：** 删除下方第一条黄鱼时，SAMTok 显式输出能清理目标区域并保持珊瑚和红鱼；baseline 更容易留下目标或出现局部残影。

### 0342｜compbench｜remove

**指令：** remove the rightmost sheep

![案例 0342 对比](<assets/independent_case_study/figures/0342_remove_compbench.jpg>)

**独立观察：** 删除最右羊的显式 final 基本保持羊群和雪地，局部边界略软；baseline 仍有目标羊残留。SAMTok 的区域外保真较好。

### 0378｜compbench｜remove

**指令：** remove the leftmost duck in the group of ducks

![案例 0378 对比](<assets/independent_case_study/figures/0378_remove_compbench.jpg>)

**独立观察：** 显式输入下 final 能删除左侧鸭并保留右侧鸭群；text-only 会扩大擦除范围。baseline 的定位也不稳定，SAMTok 的显式 token 更可靠。

### 0414｜compbench｜remove

**指令：** remove the leftmost cow

![案例 0414 对比](<assets/independent_case_study/figures/0414_remove_compbench.jpg>)

**独立观察：** 显式 final 能清理左侧牛所在区域，同时保留道路、雪和行人；baseline 在部分 setting 仍残留牛。属于 remove 的清晰正例。

### 0451｜compbench｜remove

**指令：** remove the leftmost panda

![案例 0451 对比](<assets/independent_case_study/figures/0451_remove_compbench.jpg>)

**独立观察：** 删除左下熊猫时，SAMTok 显式结果保留上方熊猫和篮筐，背景补全相对自然；text-only 会把树枝区域擦得更宽。

### 0470｜compbench｜replace

**指令：** replace the person on the right with a person in white

![案例 0470 对比](<assets/independent_case_study/figures/0470_replace_compbench.jpg>)

**独立观察：** 替换右侧骑行者为白衣人时，SAMTok 四种 setting 都能把目标人物改为白衣并保持道路结构；baseline 也能完成，但 SAMTok 的显式边界更干净。

### 0475｜compbench｜replace

**指令：** replace the third giraffe from the right with background

![案例 0475 对比](<assets/independent_case_study/figures/0475_replace_compbench.jpg>)

**独立观察：** 把第三只长颈鹿替换为背景时，SAMTok 显式 final 的目标区域基本干净，step 中的白色鬼影在 final 减少；baseline 显式结果更容易留下轮廓或 marker。

### 0480｜compbench｜replace

**指令：** replace the leftmost rabbit with a small table model

![案例 0480 对比](<assets/independent_case_study/figures/0480_replace_compbench.jpg>)

**独立观察：** 兔子替换为小桌子时，SAMTok 显式结果能放置桌子，位置正确但桌型/颜色与 target 不同；baseline 往往桌子过大或同时重绘旁边兔子。

### 0486｜compbench｜replace

**指令：** replace the rightmost car with a black car moving to the right

![案例 0486 对比](<assets/independent_case_study/figures/0486_replace_compbench.jpg>)

**独立观察：** 右侧车辆改为黑车是稳定 replace。SAMTok 显式 final 的车辆位置、道路和树木保持较好，baseline 也能完成，差异主要在车辆细节。

### 0491｜compbench｜replace

**指令：** replace the red motorcycle on the right with a black motorcycle

![案例 0491 对比](<assets/independent_case_study/figures/0491_replace_compbench.jpg>)

**独立观察：** 右侧摩托改黑色时，SAMTok 的显式 final 接近 target，未明显改变左侧摩托；baseline 也能完成但有时形状更大。

### 0513｜compbench｜add

**指令：** add a chicken in the middle and a chicken on the bottom left

![案例 0513 对比](<assets/independent_case_study/figures/0513_add_compbench.jpg>)

**独立观察：** 双实例新增是相对成功的 add：显式 setting 能在中部和左下放置两只鸡，目标数量大体正确；baseline 经常生成过大的鸡或保留 locator。SAMTok 位置更受控，但颜色、姿态和边缘仍有偏差。

### 0531｜compbench｜remove

**指令：** remove the chicken in the middle and the chicken on the bottom left

![案例 0531 对比](<assets/independent_case_study/figures/0531_remove_compbench.jpg>)

**独立观察：** 双目标删除的显式 final 能去掉中部和左下鸡，但 text-only/step 会出现大块模糊或新白鸡，且部分 final 对剩余鸡的数量不够稳定。多目标 remove 比单目标更容易过度编辑。

### 0532｜humanedit｜replace

**指令：** Replace the black hat of the big doll in the middle with a Santa hat

![案例 0532 对比](<assets/independent_case_study/figures/0532_replace_humanedit.jpg>)

**独立观察：** Santa 帽语义完成，但 SAMTok final 的帽子偏大并压住人偶脸部；baseline 的帽形相对接近 target。局部属性替换需要更细的边界和尺度控制。

### 0533｜humanedit｜add

**指令：** Add a white cat next to the gray cat.

![案例 0533 对比](<assets/independent_case_study/figures/0533_add_humanedit.jpg>)

**独立观察：** 显式 mask/box/point 能在灰猫旁新增白猫，位置和数量较好；baseline 可完成但容易保留红框。SAMTok 在这个 add 小物体案例中表现出可用的空间控制。

### 0535｜humanedit｜add

**指令：** Add a white strawberry to the middle of several red strawberries.

![案例 0535 对比](<assets/independent_case_study/figures/0535_add_humanedit.jpg>)

**独立观察：** 白草莓 add 是明显失败：SAMTok final 多数只生成白色团块或改变原草莓，没有形成清晰的白草莓；baseline 更接近目标形状但仍有红框。

### 0536｜humanedit｜remove

**指令：** Remove the little boy next to the little girl.

![案例 0536 对比](<assets/independent_case_study/figures/0536_remove_humanedit.jpg>)

**独立观察：** 删除小男孩时，显式 final 能保留小女孩和背景，区域外结构自然；baseline 部分 setting 未完全删除。

### 0539｜humanedit｜replace

**指令：** Replace the snowflake pattern on the second block from the left with the letter Y.

![案例 0539 对比](<assets/independent_case_study/figures/0539_replace_humanedit.jpg>)

**独立观察：** 把第二块雪花改为 Y 时，SAMTok 显式 final 能准确改变字母且保持其他方块；baseline 也能完成。细粒度文字替换在本例可用。

### 0544｜humanedit｜add

**指令：** Add a bee in the center of the flower in the middle of the picture

![案例 0544 对比](<assets/independent_case_study/figures/0544_add_humanedit.jpg>)

**独立观察：** 在花中心新增蜜蜂时，SAMTok 多数 setting 不生成蜜蜂或只留下很小模糊斑，baseline 至少生成了蜜蜂但尺寸偏大。新增细小主体是当前弱项。

### 0545｜humanedit｜remove

**指令：** Remove all the people behind the man in the center of the picture

![案例 0545 对比](<assets/independent_case_study/figures/0545_remove_humanedit.jpg>)

**独立观察：** 删除中心人物后方的人时，SAMTok 显式 final 能保留前景人物并修复背景；baseline 更容易保留后方人。

### 0546｜humanedit｜replace

**指令：** Replace the hat worn by the person on the far left with a white hat.

![案例 0546 对比](<assets/independent_case_study/figures/0546_replace_humanedit.jpg>)

**独立观察：** 把最左人物帽子改白时，SAMTok final 的显式输出准确修改目标帽子，其他人物基本保持；baseline 也能完成。

### 0551｜humanedit｜replace

**指令：** Replace the empty plate on the left side of the table with a plate with food.

![案例 0551 对比](<assets/independent_case_study/figures/0551_replace_humanedit.jpg>)

**独立观察：** 把左侧空盘换成有食物的盘子时，SAMTok 显式 final 放置了食物盘，位置对但玻璃杯和桌面有轻微联动变化；baseline 改动更宽。

### 0552｜humanedit｜remove

**指令：** Remove the tallest tree in the middle of the forest.

![案例 0552 对比](<assets/independent_case_study/figures/0552_remove_humanedit.jpg>)

**独立观察：** 删除森林中央最高树是非常清晰的成功例，SAMTok final 的显式输出与 target 接近，baseline 多数仍保留树。

### 0555｜humanedit｜remove

**指令：** Remove all but the fallen volleyball player

![案例 0555 对比](<assets/independent_case_study/figures/0555_remove_humanedit.jpg>)

**独立观察：** 要求只保留倒地排球运动员。SAMTok 显式结果比 baseline 更接近，但仍常保留右侧站立人物或留下擦除痕迹；多对象保留/删除边界仍不完整。

### 0556｜mirage｜mixed

**指令：** Change the color of the shell of the leftmost turtle to red, and add some snow onto the shell of the rightmost turtle.

![案例 0556 对比](<assets/independent_case_study/figures/0556_mixed_mirage.jpg>)

**独立观察：** 显式 token 能把左龟变红、右龟加雪，目标对象绑定比 baseline（多只龟变红）更准确；雪会溢出到路面，说明区域外保真仍有限。

### 0557｜mirage｜replace

**指令：** Change the color of the middle car's headlights to red, and change the color of the rightmost car's logo to blue.

![案例 0557 对比](<assets/independent_case_study/figures/0557_replace_mirage.jpg>)

**独立观察：** 车辆颜色属性的 final 能部分改变中间车灯，但右车 logo 的蓝色不稳定，属于 under-edit；baseline 更激进，容易把多辆车一起改色。

### 0579｜mirage｜mixed

**指令：** Add a bunch of red roses to the middle person's hand, and change the color of the right person's pants to blue.

![案例 0579 对比](<assets/independent_case_study/figures/0579_mixed_mirage.jpg>)

**独立观察：** final 能放置红玫瑰并部分改变右侧裤子，但裤色/位置不总是完整；baseline 也常把多个区域一起染色。SAMTok 的多 span 空间控制较好但属性完成不足。

### 0581｜mirage｜remove

**指令：** Remove the middle bird's wings.

![案例 0581 对比](<assets/independent_case_study/figures/0581_remove_mirage.jpg>)

**独立观察：** 删除中间鸟的翅膀时，SAMTok 显式 final 经常直接删除整只鸟，而不是保留无翅身体；这是局部结构编辑失败，baseline 同样不稳定。

### 0585｜mirage｜replace

**指令：** Change the color of the left dog's eyes to blue, and change the right dog's fur texture to rough.

![案例 0585 对比](<assets/independent_case_study/figures/0585_replace_mirage.jpg>)

**独立观察：** 左狗眼睛变蓝、右狗毛变粗时，SAMTok 显式 final 常把多只狗整体染蓝，目标绑定失败；baseline 也会扩大编辑，但相对更接近局部属性。

### 0590｜mirage｜mixed

**指令：** Change the color of the middle person's shirt to yellow, and add some lava onto the left person's shirt.

![案例 0590 对比](<assets/independent_case_study/figures/0590_mixed_mirage.jpg>)

**独立观察：** 中间衣服变黄、左衣服加熔岩时，SAMTok 显式 final 能分别命中两个主体，整体比 baseline 的多人物过度编辑更克制；熔岩边界和衣服纹理仍偏粗。

### 0605｜mirage｜mixed

**指令：** Change the color of the middle person's shirt to green, and add some snow onto the left man's shoulder.

![案例 0605 对比](<assets/independent_case_study/figures/0605_mixed_mirage.jpg>)

**独立观察：** 中间衣服改绿的空间绑定较好，左肩雪有时很弱或缺失；baseline 容易把多个衣服一起改绿。

### 0608｜mirage｜add

**指令：** Add a blue scarf to the first cat from the left, and add a red scarf to the second cat from the right.

![案例 0608 对比](<assets/independent_case_study/figures/0608_add_mirage.jpg>)

**独立观察：** 给两只猫加不同颜色围巾时，SAMTok final 多数不生成清晰围巾，baseline 显式结果反而能呈现蓝/红围巾。多主体、多颜色 add 明显困难。

### 0611｜mirage｜add

**指令：** Add some snow onto the head of the second horse from the right, and add some lava onto the head of the first horse from the right.

![案例 0611 对比](<assets/independent_case_study/figures/0611_add_mirage.jpg>)

**独立观察：** 雪和熔岩都能被生成，但 SAMTok 常把效果放到相邻马或扩大到头顶之外；baseline 更容易全体加火焰。SAMTok 更克制但目标身份仍不稳定。

### 0615｜mirage｜mixed

**指令：** Add a scarf to the neck of the first dog from the right, and remove the red scarf on the neck of the second dog from the right.

![案例 0615 对比](<assets/independent_case_study/figures/0615_mixed_mirage.jpg>)

**独立观察：** 同时新增右侧狗围巾并删除另一只围巾时，SAMTok final 往往两项都不完整，保持原图；复杂 add+remove 组合需要更强的 clause 绑定。

### 0622｜mirage｜mixed

**指令：** Add a rose pattern onto the chest of the first sheep from the left, and change the color of the ears of the first sheep from the right to pink.

![案例 0622 对比](<assets/independent_case_study/figures/0622_mixed_mirage.jpg>)

**独立观察：** 显式 final 能在左羊胸口形成玫瑰、右羊耳朵变粉，且比 baseline 给所有羊加花/粉耳更局部；这是 mixed 属性编辑的较好例子。

### 0627｜mirage｜replace

**指令：** Change the color of the shirt of the first person from the left to blue, and change the material of the pants of the second person from the right to rubber.

![案例 0627 对比](<assets/independent_case_study/figures/0627_replace_mirage.jpg>)

**独立观察：** 第一人衣服变蓝的目标通常能完成，第二人裤子材质变化较难从视觉上稳定呈现；SAMTok 比 baseline 更少改动其他人，但材质编辑仍缺乏可见一致性。

### 0628｜mirage｜mixed

**指令：** Change the screen display of the second machine from the right to a beach vacation scene, and add some lava onto the second machine from the left.

![案例 0628 对比](<assets/independent_case_study/figures/0628_mixed_mirage.jpg>)

**独立观察：** 机器屏幕改海滩、另一台加熔岩的复杂 mixed 任务中，SAMTok final 常只生成大块熔岩，屏幕没有按要求变化；baseline 也有大范围火焰但部分屏幕变化更明显。

### 0651｜mirage｜remove

**指令：** Remove the first snowmobile from the left, and remove the dirt on the front of the third snowmobile from the left.

![案例 0651 对比](<assets/independent_case_study/figures/0651_remove_mirage.jpg>)

**独立观察：** 删除左侧雪地车和第三车前污渍时，SAMTok 显式 final 多数删掉目标车并保留右侧车辆；部分 setting 会把道路/水雾一起抹掉，污渍清理不稳定。

### 0654｜mirage｜replace

**指令：** Change the material of the lanyard of the second person from the left to ceramic, and change the color of the hair of the fifth person from the left to white.

![案例 0654 对比](<assets/independent_case_study/figures/0654_replace_mirage.jpg>)

**独立观察：** 第五人的头发变白、第二人的挂绳材质变化在显式 final 中较可见，其他人基本保持；属于细粒度属性编辑的相对成功例。

### 0655｜mirage｜mixed

**指令：** Change the color of the eyes of the first sparrow from the left to green, and remove the white down feathers on the third sparrow from the left.

![案例 0655 对比](<assets/independent_case_study/figures/0655_mixed_mirage.jpg>)

**独立观察：** 要求第一只麻雀眼睛变绿并删除第三只白羽时，SAMTok 往往把多只鸟变绿或直接抹掉鸟，局部属性和局部删除同时出现时失败明显。

## 3. 人工案例归纳（不参考自动评分）

本节覆盖前文全部 48 个 case。每个 case 的完整输入、六列对比图和逐案例观察保留在第 2 节；这里把结论压缩为一句，并按人工视觉判断分为两组。“做得好或基本可用”包括明显优于 baseline 以及结果可用但仍有局部缺陷的样本。

### 3.1 做得好或基本可用

| case | 数据集 / 类型 | 精简结论 |
|---|---|---|
| 0036 | compbench / add | 显式区域能在指定位置放置斑马，位置绑定优于 baseline；姿态仍不稳定。 |
| 0109 | compbench / add | 黑鸭位置和尺度接近目标，显式区域比 text-only 更稳。 |
| 0145 | compbench / add | 右侧新增白飞机并保留原飞机，区域控制优于 baseline。 |
| 0233 | compbench / remove | 能删左鱼并保留右橙鱼，显式区域减少错绑。 |
| 0269 | compbench / remove | 显式区域基本只删左下鸡；text-only 有重排或引入白鸡。 |
| 0306 | compbench / remove | 黄鱼清理较完整，珊瑚和红鱼保留较好。 |
| 0342 | compbench / remove | 删除右羊并保留羊群和雪地，边界略软但主体关系正确。 |
| 0378 | compbench / remove | 删除左鸭、保留右鸭群，显式区域明显优于 text-only。 |
| 0414 | compbench / remove | 删除左牛并保留道路、雪和行人，baseline 有残留。 |
| 0451 | compbench / remove | 左熊猫删除和背景补全较自然，text-only 擦除范围偏宽。 |
| 0470 | compbench / replace | 骑行者衣服替换成功，显式边界较干净。 |
| 0475 | compbench / replace | 长颈鹿替换后的背景更干净，baseline 有轮廓或 marker 残留。 |
| 0480 | compbench / replace | 兔子位置保持较好，桌型和颜色仍有偏差。 |
| 0486 | compbench / replace | 右车改黑较稳定，场景空间关系保持好。 |
| 0491 | compbench / replace | 右摩托颜色修改较明确，左摩托基本未被误改。 |
| 0513 | compbench / add | 双鸡基本能按两个区域放置，位置受控；颜色、姿态和边缘仍有差异。 |
| 0531 | compbench / remove | 双鸡删除基本完成，但多目标边界和数量稳定性一般。 |
| 0532 | humanedit / replace | 黑帽替换为 Santa 帽语义完成，帽子尺度略大。 |
| 0533 | humanedit / add | 灰猫旁新增白猫，位置和数量较好；baseline 有红框痕迹。 |
| 0536 | humanedit / remove | 删除小男孩并保留小女孩和背景，baseline 删除不完整。 |
| 0539 | humanedit / replace | 雪花图案改为 Y，显式区域结果可用。 |
| 0545 | humanedit / remove | 后方人物删除、前景保留较好，baseline 更易残留。 |
| 0546 | humanedit / replace | 左人物帽子改白，其他人物基本保持。 |
| 0551 | humanedit / replace | 空盘变为有食物的盘，位置正确但有轻微桌面联动。 |
| 0552 | humanedit / remove | 删除中央最高的树，final 接近目标，baseline 多有残留。 |
| 0555 | humanedit / remove | 只保留倒地排球员时比 baseline 更接近，但站立人物和擦除痕仍可能残留。 |
| 0556 | mirage / mixed | 左龟变红、右龟加雪的区域绑定较好，雪有轻微溢出。 |
| 0590 | mirage / mixed | 中衣变黄、左衣加熔岩均能命中目标，较 baseline 克制；边界偏粗。 |
| 0605 | mirage / mixed | 中衣变绿的空间绑定好，左肩加雪较弱；baseline 更易扩散到整件衣服。 |
| 0622 | mirage / mixed | 左羊胸口和右羊耳的局部编辑可见，优于大范围扩散。 |
| 0627 | mirage / replace | 第一人衣服变蓝较好，第二人裤子材质不稳定，但误伤少。 |
| 0654 | mirage / replace | 第五人头发变白、第二人挂绳变化可见，其他人基本保持。 |

### 3.2 做得不好或明显薄弱

| case | 数据集 / 类型 | 精简结论 |
|---|---|---|
| 0000 | compbench / add | 新增鱼不稳定，姿态和数量常不对；baseline 更常生成可识别的鱼。 |
| 0073 | compbench / add | 位置大致正确但对象偏小或数量不足，baseline 更积极。 |
| 0181 | compbench / add | 新增灰猫模糊且尺度错误，baseline 更接近目标。 |
| 0218 | compbench / add | 常不新增或只生成灰雾，baseline 的 mask/box 结果更接近。 |
| 0535 | humanedit / add | 白草莓常变成白团块或改动原草莓，baseline 形状更完整。 |
| 0544 | humanedit / add | 花中心蜜蜂常不生成或模糊，baseline 至少能生成目标。 |
| 0557 | mirage / replace | 中车灯局部变红、右车 logo 变化不稳，baseline 更激进但也会误改。 |
| 0579 | mirage / mixed | 红玫瑰生成和裤色修改均不完整，属性编辑容易扩散。 |
| 0581 | mirage / remove | 去掉鸟翅常退化为删除整只鸟，局部结构编辑失败。 |
| 0585 | mirage / replace | 只改局部眼睛或毛发时，常把多只狗整体染蓝。 |
| 0608 | mirage / add | 两只猫的围巾常不清晰，baseline 的显式区域结果反而更完整。 |
| 0611 | mirage / add | 雪或熔岩会扩散到相邻马，或扩大为整片效果。 |
| 0615 | mirage / mixed | 加右狗围巾和删另一条围巾常只完成一项，或两项都不完整。 |
| 0628 | mirage / mixed | 常只生成大块熔岩，机器屏幕没有按要求变化。 |
| 0651 | mirage / remove | 删除车辆时道路和水雾也可能被抹掉，污渍清理不稳定。 |
| 0655 | mirage / mixed | 多只鸟被同时变绿或被抹除，局部属性和删除均不稳定。 |

### 3.3 综合结论

- **主要提升：** 显式 mask、box、point 下，remove 的目标绑定和区域外保持最稳定；单目标 replace 通常能命中指定对象；多实例场景中的错绑明显减少。
- **主要不足：** add、小目标、姿态约束、局部部件或材质编辑仍容易 under-edit、尺度错误、整物体重绘或产生 mask 外扩散；复杂 mixed 常只完成一条指令。
- **与 baseline 的关系：** SAMTok 更擅长空间绑定和保留非目标区域；baseline 在 add 和细节生成上更积极，但也更容易残留 marker、错绑或过度修改。当前结果支持“改善区域控制”，不支持“所有编辑类型全面超过 baseline”。
- **后续重点：** 提高新增实体和小目标监督，强化多 clause 的逐区域对应，增加局部属性和区域外重建约束，并继续检查 text-only 与显式区域输入的一致性。

## 4. 结论

48 个案例的独立观察表明，当前方法最可靠的收益是“编辑指定对象并尽量保留其他区域”，其中 remove 和单目标 replace 表现最好。add、细小主体、局部结构属性及复杂 mixed 仍是主要瓶颈；失败通常表现为目标未生成、只完成部分指令、局部编辑扩大到整物体，或背景被一并重绘。
