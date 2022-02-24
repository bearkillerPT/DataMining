<!DOCTYPE html>
<html>
<head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>DataMiningProject</title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>




<style type="text/css">
    pre { line-height: 125%; }
td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
.highlight .hll { background-color: var(--jp-cell-editor-active-background) }
.highlight { background: var(--jp-cell-editor-background); color: var(--jp-mirror-editor-variable-color) }
.highlight .c { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment */
.highlight .err { color: var(--jp-mirror-editor-error-color) } /* Error */
.highlight .k { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword */
.highlight .o { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator */
.highlight .p { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation */
.highlight .ch { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Multiline */
.highlight .cp { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Preproc */
.highlight .cpf { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Single */
.highlight .cs { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Special */
.highlight .kc { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Pseudo */
.highlight .kr { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Type */
.highlight .m { color: var(--jp-mirror-editor-number-color) } /* Literal.Number */
.highlight .s { color: var(--jp-mirror-editor-string-color) } /* Literal.String */
.highlight .ow { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator.Word */
.highlight .w { color: var(--jp-mirror-editor-variable-color) } /* Text.Whitespace */
.highlight .mb { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Bin */
.highlight .mf { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Float */
.highlight .mh { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Hex */
.highlight .mi { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer */
.highlight .mo { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Oct */
.highlight .sa { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Affix */
.highlight .sb { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Backtick */
.highlight .sc { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Char */
.highlight .dl { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Delimiter */
.highlight .sd { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Doc */
.highlight .s2 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Double */
.highlight .se { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Escape */
.highlight .sh { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Heredoc */
.highlight .si { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Interpol */
.highlight .sx { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Other */
.highlight .sr { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Regex */
.highlight .s1 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Single */
.highlight .ss { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Symbol */
.highlight .il { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer.Long */
  </style>



<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
 * Mozilla scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */
[data-jp-theme-scrollbars='true'] {
  scrollbar-color: rgb(var(--jp-scrollbar-thumb-color))
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar. These selectors
 * will match lower in the tree, and so will override the above */
[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
}

/* tiny scrollbar */

.jp-scrollbar-tiny {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
  scrollbar-width: thin;
}

/*
 * Webkit scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar,
[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-corner {
  background: var(--jp-scrollbar-background-color);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-thumb {
  background: rgb(var(--jp-scrollbar-thumb-color));
  border: var(--jp-scrollbar-thumb-margin) solid transparent;
  background-clip: content-box;
  border-radius: var(--jp-scrollbar-thumb-radius);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-track:horizontal {
  border-left: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
  border-right: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-track:vertical {
  border-top: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
  border-bottom: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar */

[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar::-webkit-scrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar::-webkit-scrollbar,
[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-corner,
[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-corner {
  background-color: transparent;
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-thumb,
[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
  border: var(--jp-scrollbar-thumb-margin) solid transparent;
  background-clip: content-box;
  border-radius: var(--jp-scrollbar-thumb-radius);
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-track:horizontal {
  border-left: var(--jp-scrollbar-endpad) solid transparent;
  border-right: var(--jp-scrollbar-endpad) solid transparent;
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-track:vertical {
  border-top: var(--jp-scrollbar-endpad) solid transparent;
  border-bottom: var(--jp-scrollbar-endpad) solid transparent;
}

/* tiny scrollbar */

.jp-scrollbar-tiny::-webkit-scrollbar,
.jp-scrollbar-tiny::-webkit-scrollbar-corner {
  background-color: transparent;
  height: 4px;
  width: 4px;
}

.jp-scrollbar-tiny::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:horizontal {
  border-left: 0px solid transparent;
  border-right: 0px solid transparent;
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:vertical {
  border-top: 0px solid transparent;
  border-bottom: 0px solid transparent;
}

/*
 * Phosphor
 */

.lm-ScrollBar[data-orientation='horizontal'] {
  min-height: 16px;
  max-height: 16px;
  min-width: 45px;
  border-top: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] {
  min-width: 16px;
  max-width: 16px;
  min-height: 45px;
  border-left: 1px solid #a0a0a0;
}

.lm-ScrollBar-button {
  background-color: #f0f0f0;
  background-position: center center;
  min-height: 15px;
  max-height: 15px;
  min-width: 15px;
  max-width: 15px;
}

.lm-ScrollBar-button:hover {
  background-color: #dadada;
}

.lm-ScrollBar-button.lm-mod-active {
  background-color: #cdcdcd;
}

.lm-ScrollBar-track {
  background: #f0f0f0;
}

.lm-ScrollBar-thumb {
  background: #cdcdcd;
}

.lm-ScrollBar-thumb:hover {
  background: #bababa;
}

.lm-ScrollBar-thumb.lm-mod-active {
  background: #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal'] .lm-ScrollBar-thumb {
  height: 100%;
  min-width: 15px;
  border-left: 1px solid #a0a0a0;
  border-right: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] .lm-ScrollBar-thumb {
  width: 100%;
  min-height: 15px;
  border-top: 1px solid #a0a0a0;
  border-bottom: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-left);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-right);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-up);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-down);
  background-size: 17px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-Widget, /* </DEPRECATED> */
.lm-Widget {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  cursor: default;
}


/* <DEPRECATED> */ .p-Widget.p-mod-hidden, /* </DEPRECATED> */
.lm-Widget.lm-mod-hidden {
  display: none !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-CommandPalette, /* </DEPRECATED> */
.lm-CommandPalette {
  display: flex;
  flex-direction: column;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-CommandPalette-search, /* </DEPRECATED> */
.lm-CommandPalette-search {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-content, /* </DEPRECATED> */
.lm-CommandPalette-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  min-height: 0;
  overflow: auto;
  list-style-type: none;
}


/* <DEPRECATED> */ .p-CommandPalette-header, /* </DEPRECATED> */
.lm-CommandPalette-header {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}


/* <DEPRECATED> */ .p-CommandPalette-item, /* </DEPRECATED> */
.lm-CommandPalette-item {
  display: flex;
  flex-direction: row;
}


/* <DEPRECATED> */ .p-CommandPalette-itemIcon, /* </DEPRECATED> */
.lm-CommandPalette-itemIcon {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-itemContent, /* </DEPRECATED> */
.lm-CommandPalette-itemContent {
  flex: 1 1 auto;
  overflow: hidden;
}


/* <DEPRECATED> */ .p-CommandPalette-itemShortcut, /* </DEPRECATED> */
.lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-itemLabel, /* </DEPRECATED> */
.lm-CommandPalette-itemLabel {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-close-icon {
	border:1px solid transparent;
  background-color: transparent;
  position: absolute;
	z-index:1;
	right:3%;
	top: 0;
	bottom: 0;
	margin: auto;
	padding: 7px 0;
	display: none;
	vertical-align: middle;
  outline: 0;
  cursor: pointer;
}
.lm-close-icon:after {
	content: "X";
	display: block;
	width: 15px;
	height: 15px;
	text-align: center;
	color:#000;
	font-weight: normal;
	font-size: 12px;
	cursor: pointer;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-DockPanel, /* </DEPRECATED> */
.lm-DockPanel {
  z-index: 0;
}


/* <DEPRECATED> */ .p-DockPanel-widget, /* </DEPRECATED> */
.lm-DockPanel-widget {
  z-index: 0;
}


/* <DEPRECATED> */ .p-DockPanel-tabBar, /* </DEPRECATED> */
.lm-DockPanel-tabBar {
  z-index: 1;
}


/* <DEPRECATED> */ .p-DockPanel-handle, /* </DEPRECATED> */
.lm-DockPanel-handle {
  z-index: 2;
}


/* <DEPRECATED> */ .p-DockPanel-handle.p-mod-hidden, /* </DEPRECATED> */
.lm-DockPanel-handle.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-DockPanel-handle:after, /* </DEPRECATED> */
.lm-DockPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='horizontal'],
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='horizontal'] {
  cursor: ew-resize;
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='vertical'],
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='vertical'] {
  cursor: ns-resize;
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='horizontal']:after,
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='horizontal']:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='vertical']:after,
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='vertical']:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}


/* <DEPRECATED> */ .p-DockPanel-overlay, /* </DEPRECATED> */
.lm-DockPanel-overlay {
  z-index: 3;
  box-sizing: border-box;
  pointer-events: none;
}


/* <DEPRECATED> */ .p-DockPanel-overlay.p-mod-hidden, /* </DEPRECATED> */
.lm-DockPanel-overlay.lm-mod-hidden {
  display: none !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-Menu, /* </DEPRECATED> */
.lm-Menu {
  z-index: 10000;
  position: absolute;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: auto;
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-Menu-content, /* </DEPRECATED> */
.lm-Menu-content {
  margin: 0;
  padding: 0;
  display: table;
  list-style-type: none;
}


/* <DEPRECATED> */ .p-Menu-item, /* </DEPRECATED> */
.lm-Menu-item {
  display: table-row;
}


/* <DEPRECATED> */
.p-Menu-item.p-mod-hidden,
.p-Menu-item.p-mod-collapsed,
/* </DEPRECATED> */
.lm-Menu-item.lm-mod-hidden,
.lm-Menu-item.lm-mod-collapsed {
  display: none !important;
}


/* <DEPRECATED> */
.p-Menu-itemIcon,
.p-Menu-itemSubmenuIcon,
/* </DEPRECATED> */
.lm-Menu-itemIcon,
.lm-Menu-itemSubmenuIcon {
  display: table-cell;
  text-align: center;
}


/* <DEPRECATED> */ .p-Menu-itemLabel, /* </DEPRECATED> */
.lm-Menu-itemLabel {
  display: table-cell;
  text-align: left;
}


/* <DEPRECATED> */ .p-Menu-itemShortcut, /* </DEPRECATED> */
.lm-Menu-itemShortcut {
  display: table-cell;
  text-align: right;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-MenuBar, /* </DEPRECATED> */
.lm-MenuBar {
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-MenuBar-content, /* </DEPRECATED> */
.lm-MenuBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: row;
  list-style-type: none;
}


/* <DEPRECATED> */ .p--MenuBar-item, /* </DEPRECATED> */
.lm-MenuBar-item {
  box-sizing: border-box;
}


/* <DEPRECATED> */
.p-MenuBar-itemIcon,
.p-MenuBar-itemLabel,
/* </DEPRECATED> */
.lm-MenuBar-itemIcon,
.lm-MenuBar-itemLabel {
  display: inline-block;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-ScrollBar, /* </DEPRECATED> */
.lm-ScrollBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */
.p-ScrollBar[data-orientation='horizontal'],
/* </DEPRECATED> */
.lm-ScrollBar[data-orientation='horizontal'] {
  flex-direction: row;
}


/* <DEPRECATED> */
.p-ScrollBar[data-orientation='vertical'],
/* </DEPRECATED> */
.lm-ScrollBar[data-orientation='vertical'] {
  flex-direction: column;
}


/* <DEPRECATED> */ .p-ScrollBar-button, /* </DEPRECATED> */
.lm-ScrollBar-button {
  box-sizing: border-box;
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-ScrollBar-track, /* </DEPRECATED> */
.lm-ScrollBar-track {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  flex: 1 1 auto;
}


/* <DEPRECATED> */ .p-ScrollBar-thumb, /* </DEPRECATED> */
.lm-ScrollBar-thumb {
  box-sizing: border-box;
  position: absolute;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-SplitPanel-child, /* </DEPRECATED> */
.lm-SplitPanel-child {
  z-index: 0;
}


/* <DEPRECATED> */ .p-SplitPanel-handle, /* </DEPRECATED> */
.lm-SplitPanel-handle {
  z-index: 1;
}


/* <DEPRECATED> */ .p-SplitPanel-handle.p-mod-hidden, /* </DEPRECATED> */
.lm-SplitPanel-handle.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-SplitPanel-handle:after, /* </DEPRECATED> */
.lm-SplitPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle {
  cursor: ew-resize;
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle {
  cursor: ns-resize;
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle:after,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle:after,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-TabBar, /* </DEPRECATED> */
.lm-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-TabBar[data-orientation='horizontal'], /* </DEPRECATED> */
.lm-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
}


/* <DEPRECATED> */ .p-TabBar[data-orientation='vertical'], /* </DEPRECATED> */
.lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
}


/* <DEPRECATED> */ .p-TabBar-content, /* </DEPRECATED> */
.lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}


/* <DEPRECATED> */
.p-TabBar[data-orientation='horizontal'] > .p-TabBar-content,
/* </DEPRECATED> */
.lm-TabBar[data-orientation='horizontal'] > .lm-TabBar-content {
  flex-direction: row;
}


/* <DEPRECATED> */
.p-TabBar[data-orientation='vertical'] > .p-TabBar-content,
/* </DEPRECATED> */
.lm-TabBar[data-orientation='vertical'] > .lm-TabBar-content {
  flex-direction: column;
}


/* <DEPRECATED> */ .p-TabBar-tab, /* </DEPRECATED> */
.lm-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
}


/* <DEPRECATED> */
.p-TabBar-tabIcon,
.p-TabBar-tabCloseIcon,
/* </DEPRECATED> */
.lm-TabBar-tabIcon,
.lm-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-TabBar-tabLabel, /* </DEPRECATED> */
.lm-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}


.lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing : border-box;
}


/* <DEPRECATED> */ .p-TabBar-tab.p-mod-hidden, /* </DEPRECATED> */
.lm-TabBar-tab.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-TabBar.p-mod-dragging .p-TabBar-tab, /* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
  position: relative;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging[data-orientation='horizontal'] .p-TabBar-tab,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging[data-orientation='horizontal'] .lm-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging[data-orientation='vertical'] .p-TabBar-tab,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging[data-orientation='vertical'] .lm-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging .p-TabBar-tab.p-mod-dragging,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging .lm-TabBar-tab.lm-mod-dragging {
  transition: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-TabPanel-tabBar, /* </DEPRECATED> */
.lm-TabPanel-tabBar {
  z-index: 1;
}


/* <DEPRECATED> */ .p-TabPanel-stackedPanel, /* </DEPRECATED> */
.lm-TabPanel-stackedPanel {
  z-index: 0;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

@charset "UTF-8";
html{
  -webkit-box-sizing:border-box;
          box-sizing:border-box; }

*,
*::before,
*::after{
  -webkit-box-sizing:inherit;
          box-sizing:inherit; }

body{
  font-size:14px;
  font-weight:400;
  letter-spacing:0;
  line-height:1.28581;
  text-transform:none;
  color:#182026;
  font-family:-apple-system, "BlinkMacSystemFont", "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Open Sans", "Helvetica Neue", "Icons16", sans-serif; }

p{
  margin-bottom:10px;
  margin-top:0; }

small{
  font-size:12px; }

strong{
  font-weight:600; }

::-moz-selection{
  background:rgba(125, 188, 255, 0.6); }

::selection{
  background:rgba(125, 188, 255, 0.6); }
.bp3-heading{
  color:#182026;
  font-weight:600;
  margin:0 0 10px;
  padding:0; }
  .bp3-dark .bp3-heading{
    color:#f5f8fa; }

h1.bp3-heading, .bp3-running-text h1{
  font-size:36px;
  line-height:40px; }

h2.bp3-heading, .bp3-running-text h2{
  font-size:28px;
  line-height:32px; }

h3.bp3-heading, .bp3-running-text h3{
  font-size:22px;
  line-height:25px; }

h4.bp3-heading, .bp3-running-text h4{
  font-size:18px;
  line-height:21px; }

h5.bp3-heading, .bp3-running-text h5{
  font-size:16px;
  line-height:19px; }

h6.bp3-heading, .bp3-running-text h6{
  font-size:14px;
  line-height:16px; }
.bp3-ui-text{
  font-size:14px;
  font-weight:400;
  letter-spacing:0;
  line-height:1.28581;
  text-transform:none; }

.bp3-monospace-text{
  font-family:monospace;
  text-transform:none; }

.bp3-text-muted{
  color:#5c7080; }
  .bp3-dark .bp3-text-muted{
    color:#a7b6c2; }

.bp3-text-disabled{
  color:rgba(92, 112, 128, 0.6); }
  .bp3-dark .bp3-text-disabled{
    color:rgba(167, 182, 194, 0.6); }

.bp3-text-overflow-ellipsis{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal; }
.bp3-running-text{
  font-size:14px;
  line-height:1.5; }
  .bp3-running-text h1{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h1{
      color:#f5f8fa; }
  .bp3-running-text h2{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h2{
      color:#f5f8fa; }
  .bp3-running-text h3{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h3{
      color:#f5f8fa; }
  .bp3-running-text h4{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h4{
      color:#f5f8fa; }
  .bp3-running-text h5{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h5{
      color:#f5f8fa; }
  .bp3-running-text h6{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h6{
      color:#f5f8fa; }
  .bp3-running-text hr{
    border:none;
    border-bottom:1px solid rgba(16, 22, 26, 0.15);
    margin:20px 0; }
    .bp3-dark .bp3-running-text hr{
      border-color:rgba(255, 255, 255, 0.15); }
  .bp3-running-text p{
    margin:0 0 10px;
    padding:0; }

.bp3-text-large{
  font-size:16px; }

.bp3-text-small{
  font-size:12px; }
a{
  color:#106ba3;
  text-decoration:none; }
  a:hover{
    color:#106ba3;
    cursor:pointer;
    text-decoration:underline; }
  a .bp3-icon, a .bp3-icon-standard, a .bp3-icon-large{
    color:inherit; }
  a code,
  .bp3-dark a code{
    color:inherit; }
  .bp3-dark a,
  .bp3-dark a:hover{
    color:#48aff0; }
    .bp3-dark a .bp3-icon, .bp3-dark a .bp3-icon-standard, .bp3-dark a .bp3-icon-large,
    .bp3-dark a:hover .bp3-icon,
    .bp3-dark a:hover .bp3-icon-standard,
    .bp3-dark a:hover .bp3-icon-large{
      color:inherit; }
.bp3-running-text code, .bp3-code{
  font-family:monospace;
  text-transform:none;
  background:rgba(255, 255, 255, 0.7);
  border-radius:3px;
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2);
  color:#5c7080;
  font-size:smaller;
  padding:2px 5px; }
  .bp3-dark .bp3-running-text code, .bp3-running-text .bp3-dark code, .bp3-dark .bp3-code{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#a7b6c2; }
  .bp3-running-text a > code, a > .bp3-code{
    color:#137cbd; }
    .bp3-dark .bp3-running-text a > code, .bp3-running-text .bp3-dark a > code, .bp3-dark a > .bp3-code{
      color:inherit; }

.bp3-running-text pre, .bp3-code-block{
  font-family:monospace;
  text-transform:none;
  background:rgba(255, 255, 255, 0.7);
  border-radius:3px;
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
  color:#182026;
  display:block;
  font-size:13px;
  line-height:1.4;
  margin:10px 0;
  padding:13px 15px 12px;
  word-break:break-all;
  word-wrap:break-word; }
  .bp3-dark .bp3-running-text pre, .bp3-running-text .bp3-dark pre, .bp3-dark .bp3-code-block{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
  .bp3-running-text pre > code, .bp3-code-block > code{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:inherit;
    font-size:inherit;
    padding:0; }

.bp3-running-text kbd, .bp3-key{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  color:#5c7080;
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  font-family:inherit;
  font-size:12px;
  height:24px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  line-height:24px;
  min-width:24px;
  padding:3px 6px;
  vertical-align:middle; }
  .bp3-running-text kbd .bp3-icon, .bp3-key .bp3-icon, .bp3-running-text kbd .bp3-icon-standard, .bp3-key .bp3-icon-standard, .bp3-running-text kbd .bp3-icon-large, .bp3-key .bp3-icon-large{
    margin-right:5px; }
  .bp3-dark .bp3-running-text kbd, .bp3-running-text .bp3-dark kbd, .bp3-dark .bp3-key{
    background:#394b59;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#a7b6c2; }
.bp3-running-text blockquote, .bp3-blockquote{
  border-left:solid 4px rgba(167, 182, 194, 0.5);
  margin:0 0 10px;
  padding:0 20px; }
  .bp3-dark .bp3-running-text blockquote, .bp3-running-text .bp3-dark blockquote, .bp3-dark .bp3-blockquote{
    border-color:rgba(115, 134, 148, 0.5); }
.bp3-running-text ul,
.bp3-running-text ol, .bp3-list{
  margin:10px 0;
  padding-left:30px; }
  .bp3-running-text ul li:not(:last-child), .bp3-running-text ol li:not(:last-child), .bp3-list li:not(:last-child){
    margin-bottom:5px; }
  .bp3-running-text ul ol, .bp3-running-text ol ol, .bp3-list ol,
  .bp3-running-text ul ul,
  .bp3-running-text ol ul,
  .bp3-list ul{
    margin-top:5px; }

.bp3-list-unstyled{
  list-style:none;
  margin:0;
  padding:0; }
  .bp3-list-unstyled li{
    padding:0; }
.bp3-rtl{
  text-align:right; }

.bp3-dark{
  color:#f5f8fa; }

:focus{
  outline:rgba(19, 124, 189, 0.6) auto 2px;
  outline-offset:2px;
  -moz-outline-radius:6px; }

.bp3-focus-disabled :focus{
  outline:none !important; }
  .bp3-focus-disabled :focus ~ .bp3-control-indicator{
    outline:none !important; }

.bp3-alert{
  max-width:400px;
  padding:20px; }

.bp3-alert-body{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }
  .bp3-alert-body .bp3-icon{
    font-size:40px;
    margin-right:20px;
    margin-top:0; }

.bp3-alert-contents{
  word-break:break-word; }

.bp3-alert-footer{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:reverse;
      -ms-flex-direction:row-reverse;
          flex-direction:row-reverse;
  margin-top:10px; }
  .bp3-alert-footer .bp3-button{
    margin-left:10px; }
.bp3-breadcrumbs{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  cursor:default;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-wrap:wrap;
      flex-wrap:wrap;
  height:30px;
  list-style:none;
  margin:0;
  padding:0; }
  .bp3-breadcrumbs > li{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex; }
    .bp3-breadcrumbs > li::after{
      background:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M10.71 7.29l-4-4a1.003 1.003 0 00-1.42 1.42L8.59 8 5.3 11.29c-.19.18-.3.43-.3.71a1.003 1.003 0 001.71.71l4-4c.18-.18.29-.43.29-.71 0-.28-.11-.53-.29-.71z' fill='%235C7080'/%3e%3c/svg%3e");
      content:"";
      display:block;
      height:16px;
      margin:0 5px;
      width:16px; }
    .bp3-breadcrumbs > li:last-of-type::after{
      display:none; }

.bp3-breadcrumb,
.bp3-breadcrumb-current,
.bp3-breadcrumbs-collapsed{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  font-size:16px; }

.bp3-breadcrumb,
.bp3-breadcrumbs-collapsed{
  color:#5c7080; }

.bp3-breadcrumb:hover{
  text-decoration:none; }

.bp3-breadcrumb.bp3-disabled{
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-breadcrumb .bp3-icon{
  margin-right:5px; }

.bp3-breadcrumb-current{
  color:inherit;
  font-weight:600; }
  .bp3-breadcrumb-current .bp3-input{
    font-size:inherit;
    font-weight:inherit;
    vertical-align:baseline; }

.bp3-breadcrumbs-collapsed{
  background:#ced9e0;
  border:none;
  border-radius:3px;
  cursor:pointer;
  margin-right:2px;
  padding:1px 5px;
  vertical-align:text-bottom; }
  .bp3-breadcrumbs-collapsed::before{
    background:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cg fill='%235C7080'%3e%3ccircle cx='2' cy='8.03' r='2'/%3e%3ccircle cx='14' cy='8.03' r='2'/%3e%3ccircle cx='8' cy='8.03' r='2'/%3e%3c/g%3e%3c/svg%3e") center no-repeat;
    content:"";
    display:block;
    height:16px;
    width:16px; }
  .bp3-breadcrumbs-collapsed:hover{
    background:#bfccd6;
    color:#182026;
    text-decoration:none; }

.bp3-dark .bp3-breadcrumb,
.bp3-dark .bp3-breadcrumbs-collapsed{
  color:#a7b6c2; }

.bp3-dark .bp3-breadcrumbs > li::after{
  color:#a7b6c2; }

.bp3-dark .bp3-breadcrumb.bp3-disabled{
  color:rgba(167, 182, 194, 0.6); }

.bp3-dark .bp3-breadcrumb-current{
  color:#f5f8fa; }

.bp3-dark .bp3-breadcrumbs-collapsed{
  background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-breadcrumbs-collapsed:hover{
    background:rgba(16, 22, 26, 0.6);
    color:#f5f8fa; }
.bp3-button{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  border:none;
  border-radius:3px;
  cursor:pointer;
  font-size:14px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  padding:5px 10px;
  text-align:left;
  vertical-align:middle;
  min-height:30px;
  min-width:30px; }
  .bp3-button > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-button > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-button::before,
  .bp3-button > *{
    margin-right:7px; }
  .bp3-button:empty::before,
  .bp3-button > :last-child{
    margin-right:0; }
  .bp3-button:empty{
    padding:0 !important; }
  .bp3-button:disabled, .bp3-button.bp3-disabled{
    cursor:not-allowed; }
  .bp3-button.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-button.bp3-align-right,
  .bp3-align-right .bp3-button{
    text-align:right; }
  .bp3-button.bp3-align-left,
  .bp3-align-left .bp3-button{
    text-align:left; }
  .bp3-button:not([class*="bp3-intent-"]){
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    color:#182026; }
    .bp3-button:not([class*="bp3-intent-"]):hover{
      background-clip:padding-box;
      background-color:#ebf1f5;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
    .bp3-button:not([class*="bp3-intent-"]):active, .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      background-color:#d8e1e8;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button:not([class*="bp3-intent-"]):disabled, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled{
      background-color:rgba(206, 217, 224, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      outline:none; }
      .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active, .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active:hover, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active:hover{
        background:rgba(206, 217, 224, 0.7); }
  .bp3-button.bp3-intent-primary{
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-primary:hover, .bp3-button.bp3-intent-primary:active, .bp3-button.bp3-intent-primary.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-primary:hover{
      background-color:#106ba3;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-primary:active, .bp3-button.bp3-intent-primary.bp3-active{
      background-color:#0e5a8a;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-primary:disabled, .bp3-button.bp3-intent-primary.bp3-disabled{
      background-color:rgba(19, 124, 189, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-success{
    background-color:#0f9960;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-success:hover, .bp3-button.bp3-intent-success:active, .bp3-button.bp3-intent-success.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-success:hover{
      background-color:#0d8050;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-success:active, .bp3-button.bp3-intent-success.bp3-active{
      background-color:#0a6640;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-success:disabled, .bp3-button.bp3-intent-success.bp3-disabled{
      background-color:rgba(15, 153, 96, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-warning{
    background-color:#d9822b;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-warning:hover, .bp3-button.bp3-intent-warning:active, .bp3-button.bp3-intent-warning.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-warning:hover{
      background-color:#bf7326;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-warning:active, .bp3-button.bp3-intent-warning.bp3-active{
      background-color:#a66321;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-warning:disabled, .bp3-button.bp3-intent-warning.bp3-disabled{
      background-color:rgba(217, 130, 43, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-danger{
    background-color:#db3737;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-danger:hover, .bp3-button.bp3-intent-danger:active, .bp3-button.bp3-intent-danger.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-danger:hover{
      background-color:#c23030;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-danger:active, .bp3-button.bp3-intent-danger.bp3-active{
      background-color:#a82a2a;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-danger:disabled, .bp3-button.bp3-intent-danger.bp3-disabled{
      background-color:rgba(219, 55, 55, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button[class*="bp3-intent-"] .bp3-button-spinner .bp3-spinner-head{
    stroke:#ffffff; }
  .bp3-button.bp3-large,
  .bp3-large .bp3-button{
    min-height:40px;
    min-width:40px;
    font-size:16px;
    padding:5px 15px; }
    .bp3-button.bp3-large::before,
    .bp3-button.bp3-large > *,
    .bp3-large .bp3-button::before,
    .bp3-large .bp3-button > *{
      margin-right:10px; }
    .bp3-button.bp3-large:empty::before,
    .bp3-button.bp3-large > :last-child,
    .bp3-large .bp3-button:empty::before,
    .bp3-large .bp3-button > :last-child{
      margin-right:0; }
  .bp3-button.bp3-small,
  .bp3-small .bp3-button{
    min-height:24px;
    min-width:24px;
    padding:0 7px; }
  .bp3-button.bp3-loading{
    position:relative; }
    .bp3-button.bp3-loading[class*="bp3-icon-"]::before{
      visibility:hidden; }
    .bp3-button.bp3-loading .bp3-button-spinner{
      margin:0;
      position:absolute; }
    .bp3-button.bp3-loading > :not(.bp3-button-spinner){
      visibility:hidden; }
  .bp3-button[class*="bp3-icon-"]::before{
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-style:normal;
    font-weight:400;
    line-height:1;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    color:#5c7080; }
  .bp3-button .bp3-icon, .bp3-button .bp3-icon-standard, .bp3-button .bp3-icon-large{
    color:#5c7080; }
    .bp3-button .bp3-icon.bp3-align-right, .bp3-button .bp3-icon-standard.bp3-align-right, .bp3-button .bp3-icon-large.bp3-align-right{
      margin-left:7px; }
  .bp3-button .bp3-icon:first-child:last-child,
  .bp3-button .bp3-spinner + .bp3-icon:last-child{
    margin:0 -7px; }
  .bp3-dark .bp3-button:not([class*="bp3-intent-"]){
    background-color:#394b59;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):hover, .bp3-dark .bp3-button:not([class*="bp3-intent-"]):active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      color:#f5f8fa; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):hover{
      background-color:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      background-color:#202b33;
      background-image:none;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):disabled, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-disabled{
      background-color:rgba(57, 75, 89, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active{
        background:rgba(57, 75, 89, 0.7); }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-button-spinner .bp3-spinner-head{
      background:rgba(16, 22, 26, 0.5);
      stroke:#8a9ba8; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"])[class*="bp3-icon-"]::before{
      color:#a7b6c2; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon, .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon-standard, .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon-large{
      color:#a7b6c2; }
  .bp3-dark .bp3-button[class*="bp3-intent-"]{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:hover{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:active, .bp3-dark .bp3-button[class*="bp3-intent-"].bp3-active{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:disabled, .bp3-dark .bp3-button[class*="bp3-intent-"].bp3-disabled{
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.3); }
    .bp3-dark .bp3-button[class*="bp3-intent-"] .bp3-button-spinner .bp3-spinner-head{
      stroke:#8a9ba8; }
  .bp3-button:disabled::before,
  .bp3-button:disabled .bp3-icon, .bp3-button:disabled .bp3-icon-standard, .bp3-button:disabled .bp3-icon-large, .bp3-button.bp3-disabled::before,
  .bp3-button.bp3-disabled .bp3-icon, .bp3-button.bp3-disabled .bp3-icon-standard, .bp3-button.bp3-disabled .bp3-icon-large, .bp3-button[class*="bp3-intent-"]::before,
  .bp3-button[class*="bp3-intent-"] .bp3-icon, .bp3-button[class*="bp3-intent-"] .bp3-icon-standard, .bp3-button[class*="bp3-intent-"] .bp3-icon-large{
    color:inherit !important; }
  .bp3-button.bp3-minimal{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none; }
    .bp3-button.bp3-minimal:hover{
      background:rgba(167, 182, 194, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026;
      text-decoration:none; }
    .bp3-button.bp3-minimal:active, .bp3-button.bp3-minimal.bp3-active{
      background:rgba(115, 134, 148, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026; }
    .bp3-button.bp3-minimal:disabled, .bp3-button.bp3-minimal:disabled:hover, .bp3-button.bp3-minimal.bp3-disabled, .bp3-button.bp3-minimal.bp3-disabled:hover{
      background:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
      .bp3-button.bp3-minimal:disabled.bp3-active, .bp3-button.bp3-minimal:disabled:hover.bp3-active, .bp3-button.bp3-minimal.bp3-disabled.bp3-active, .bp3-button.bp3-minimal.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button.bp3-minimal{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:inherit; }
      .bp3-dark .bp3-button.bp3-minimal:hover, .bp3-dark .bp3-button.bp3-minimal:active, .bp3-dark .bp3-button.bp3-minimal.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-button.bp3-minimal:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button.bp3-minimal:active, .bp3-dark .bp3-button.bp3-minimal.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button.bp3-minimal:disabled, .bp3-dark .bp3-button.bp3-minimal:disabled:hover, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled:hover{
        background:none;
        color:rgba(167, 182, 194, 0.6);
        cursor:not-allowed; }
        .bp3-dark .bp3-button.bp3-minimal:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal:disabled:hover.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:hover, .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:disabled, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-primary:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-success{
      color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:hover, .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:disabled, .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-success:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:hover, .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:disabled, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-warning:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-danger{
      color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:hover, .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:disabled, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-danger:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
  .bp3-button.bp3-outlined{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    border:1px solid rgba(24, 32, 38, 0.2);
    -webkit-box-sizing:border-box;
            box-sizing:border-box; }
    .bp3-button.bp3-outlined:hover{
      background:rgba(167, 182, 194, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026;
      text-decoration:none; }
    .bp3-button.bp3-outlined:active, .bp3-button.bp3-outlined.bp3-active{
      background:rgba(115, 134, 148, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026; }
    .bp3-button.bp3-outlined:disabled, .bp3-button.bp3-outlined:disabled:hover, .bp3-button.bp3-outlined.bp3-disabled, .bp3-button.bp3-outlined.bp3-disabled:hover{
      background:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
      .bp3-button.bp3-outlined:disabled.bp3-active, .bp3-button.bp3-outlined:disabled:hover.bp3-active, .bp3-button.bp3-outlined.bp3-disabled.bp3-active, .bp3-button.bp3-outlined.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button.bp3-outlined{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:inherit; }
      .bp3-dark .bp3-button.bp3-outlined:hover, .bp3-dark .bp3-button.bp3-outlined:active, .bp3-dark .bp3-button.bp3-outlined.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-button.bp3-outlined:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button.bp3-outlined:active, .bp3-dark .bp3-button.bp3-outlined.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button.bp3-outlined:disabled, .bp3-dark .bp3-button.bp3-outlined:disabled:hover, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled:hover{
        background:none;
        color:rgba(167, 182, 194, 0.6);
        cursor:not-allowed; }
        .bp3-dark .bp3-button.bp3-outlined:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined:disabled:hover.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:hover, .bp3-button.bp3-outlined.bp3-intent-primary:active, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:active, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-primary:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-success{
      color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:hover, .bp3-button.bp3-outlined.bp3-intent-success:active, .bp3-button.bp3-outlined.bp3-intent-success.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:active, .bp3-button.bp3-outlined.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-success:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:hover, .bp3-button.bp3-outlined.bp3-intent-warning:active, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:active, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-warning:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-danger{
      color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:hover, .bp3-button.bp3-outlined.bp3-intent-danger:active, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:active, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-danger:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
    .bp3-button.bp3-outlined:disabled, .bp3-button.bp3-outlined.bp3-disabled, .bp3-button.bp3-outlined:disabled:hover, .bp3-button.bp3-outlined.bp3-disabled:hover{
      border-color:rgba(92, 112, 128, 0.1); }
    .bp3-dark .bp3-button.bp3-outlined{
      border-color:rgba(255, 255, 255, 0.4); }
      .bp3-dark .bp3-button.bp3-outlined:disabled, .bp3-dark .bp3-button.bp3-outlined:disabled:hover, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled:hover{
        border-color:rgba(255, 255, 255, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-primary{
      border-color:rgba(16, 107, 163, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
        border-color:rgba(16, 107, 163, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary{
        border-color:rgba(72, 175, 240, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
          border-color:rgba(72, 175, 240, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-success{
      border-color:rgba(13, 128, 80, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
        border-color:rgba(13, 128, 80, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success{
        border-color:rgba(61, 204, 145, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
          border-color:rgba(61, 204, 145, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-warning{
      border-color:rgba(191, 115, 38, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
        border-color:rgba(191, 115, 38, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning{
        border-color:rgba(255, 179, 102, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
          border-color:rgba(255, 179, 102, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-danger{
      border-color:rgba(194, 48, 48, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
        border-color:rgba(194, 48, 48, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger{
        border-color:rgba(255, 115, 115, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
          border-color:rgba(255, 115, 115, 0.2); }

a.bp3-button{
  text-align:center;
  text-decoration:none;
  -webkit-transition:none;
  transition:none; }
  a.bp3-button, a.bp3-button:hover, a.bp3-button:active{
    color:#182026; }
  a.bp3-button.bp3-disabled{
    color:rgba(92, 112, 128, 0.6); }

.bp3-button-text{
  -webkit-box-flex:0;
      -ms-flex:0 1 auto;
          flex:0 1 auto; }

.bp3-button.bp3-align-left .bp3-button-text, .bp3-button.bp3-align-right .bp3-button-text,
.bp3-button-group.bp3-align-left .bp3-button-text,
.bp3-button-group.bp3-align-right .bp3-button-text{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto; }
.bp3-button-group{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex; }
  .bp3-button-group .bp3-button{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    position:relative;
    z-index:4; }
    .bp3-button-group .bp3-button:focus{
      z-index:5; }
    .bp3-button-group .bp3-button:hover{
      z-index:6; }
    .bp3-button-group .bp3-button:active, .bp3-button-group .bp3-button.bp3-active{
      z-index:7; }
    .bp3-button-group .bp3-button:disabled, .bp3-button-group .bp3-button.bp3-disabled{
      z-index:3; }
    .bp3-button-group .bp3-button[class*="bp3-intent-"]{
      z-index:9; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:focus{
        z-index:10; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:hover{
        z-index:11; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:active, .bp3-button-group .bp3-button[class*="bp3-intent-"].bp3-active{
        z-index:12; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:disabled, .bp3-button-group .bp3-button[class*="bp3-intent-"].bp3-disabled{
        z-index:8; }
  .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:first-child) .bp3-button,
  .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:first-child){
    border-bottom-left-radius:0;
    border-top-left-radius:0; }
  .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:last-child){
    border-bottom-right-radius:0;
    border-top-right-radius:0;
    margin-right:-1px; }
  .bp3-button-group.bp3-minimal .bp3-button{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none; }
    .bp3-button-group.bp3-minimal .bp3-button:hover{
      background:rgba(167, 182, 194, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026;
      text-decoration:none; }
    .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
      background:rgba(115, 134, 148, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026; }
    .bp3-button-group.bp3-minimal .bp3-button:disabled, .bp3-button-group.bp3-minimal .bp3-button:disabled:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover{
      background:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
      .bp3-button-group.bp3-minimal .bp3-button:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button:disabled:hover.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button-group.bp3-minimal .bp3-button{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:inherit; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:hover, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled:hover, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover{
        background:none;
        color:rgba(167, 182, 194, 0.6);
        cursor:not-allowed; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled:hover.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success{
      color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger{
      color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
  .bp3-button-group .bp3-popover-wrapper,
  .bp3-button-group .bp3-popover-target{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-button-group.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-button-group .bp3-button.bp3-fill,
  .bp3-button-group.bp3-fill .bp3-button:not(.bp3-fixed){
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-button-group.bp3-vertical{
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch;
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column;
    vertical-align:top; }
    .bp3-button-group.bp3-vertical.bp3-fill{
      height:100%;
      width:unset; }
    .bp3-button-group.bp3-vertical .bp3-button{
      margin-right:0 !important;
      width:100%; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:first-child .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:first-child{
      border-radius:3px 3px 0 0; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:last-child .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:last-child{
      border-radius:0 0 3px 3px; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:not(:last-child){
      margin-bottom:-1px; }
  .bp3-button-group.bp3-align-left .bp3-button{
    text-align:left; }
  .bp3-dark .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-dark .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:last-child){
    margin-right:1px; }
  .bp3-dark .bp3-button-group.bp3-vertical > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-dark .bp3-button-group.bp3-vertical > .bp3-button:not(:last-child){
    margin-bottom:1px; }
.bp3-callout{
  font-size:14px;
  line-height:1.5;
  background-color:rgba(138, 155, 168, 0.15);
  border-radius:3px;
  padding:10px 12px 9px;
  position:relative;
  width:100%; }
  .bp3-callout[class*="bp3-icon-"]{
    padding-left:40px; }
    .bp3-callout[class*="bp3-icon-"]::before{
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-style:normal;
      font-weight:400;
      line-height:1;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased;
      color:#5c7080;
      left:10px;
      position:absolute;
      top:10px; }
  .bp3-callout.bp3-callout-icon{
    padding-left:40px; }
    .bp3-callout.bp3-callout-icon > .bp3-icon:first-child{
      color:#5c7080;
      left:10px;
      position:absolute;
      top:10px; }
  .bp3-callout .bp3-heading{
    line-height:20px;
    margin-bottom:5px;
    margin-top:0; }
    .bp3-callout .bp3-heading:last-child{
      margin-bottom:0; }
  .bp3-dark .bp3-callout{
    background-color:rgba(138, 155, 168, 0.2); }
    .bp3-dark .bp3-callout[class*="bp3-icon-"]::before{
      color:#a7b6c2; }
  .bp3-callout.bp3-intent-primary{
    background-color:rgba(19, 124, 189, 0.15); }
    .bp3-callout.bp3-intent-primary[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-primary > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-primary .bp3-heading{
      color:#106ba3; }
    .bp3-dark .bp3-callout.bp3-intent-primary{
      background-color:rgba(19, 124, 189, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-primary[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-primary > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-primary .bp3-heading{
        color:#48aff0; }
  .bp3-callout.bp3-intent-success{
    background-color:rgba(15, 153, 96, 0.15); }
    .bp3-callout.bp3-intent-success[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-success > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-success .bp3-heading{
      color:#0d8050; }
    .bp3-dark .bp3-callout.bp3-intent-success{
      background-color:rgba(15, 153, 96, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-success[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-success > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-success .bp3-heading{
        color:#3dcc91; }
  .bp3-callout.bp3-intent-warning{
    background-color:rgba(217, 130, 43, 0.15); }
    .bp3-callout.bp3-intent-warning[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-warning > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-warning .bp3-heading{
      color:#bf7326; }
    .bp3-dark .bp3-callout.bp3-intent-warning{
      background-color:rgba(217, 130, 43, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-warning[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-warning > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-warning .bp3-heading{
        color:#ffb366; }
  .bp3-callout.bp3-intent-danger{
    background-color:rgba(219, 55, 55, 0.15); }
    .bp3-callout.bp3-intent-danger[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-danger > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-danger .bp3-heading{
      color:#c23030; }
    .bp3-dark .bp3-callout.bp3-intent-danger{
      background-color:rgba(219, 55, 55, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-danger[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-danger > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-danger .bp3-heading{
        color:#ff7373; }
  .bp3-running-text .bp3-callout{
    margin:20px 0; }
.bp3-card{
  background-color:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
  padding:20px;
  -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-card.bp3-dark,
  .bp3-dark .bp3-card{
    background-color:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }

.bp3-elevation-0{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }
  .bp3-elevation-0.bp3-dark,
  .bp3-dark .bp3-elevation-0{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }

.bp3-elevation-1{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-1.bp3-dark,
  .bp3-dark .bp3-elevation-1{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-elevation-2{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 1px 1px rgba(16, 22, 26, 0.2), 0 2px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 1px 1px rgba(16, 22, 26, 0.2), 0 2px 6px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-2.bp3-dark,
  .bp3-dark .bp3-elevation-2{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.4), 0 2px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.4), 0 2px 6px rgba(16, 22, 26, 0.4); }

.bp3-elevation-3{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-3.bp3-dark,
  .bp3-dark .bp3-elevation-3{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }

.bp3-elevation-4{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-4.bp3-dark,
  .bp3-dark .bp3-elevation-4{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4); }

.bp3-card.bp3-interactive:hover{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  cursor:pointer; }
  .bp3-card.bp3-interactive:hover.bp3-dark,
  .bp3-dark .bp3-card.bp3-interactive:hover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }

.bp3-card.bp3-interactive:active{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  opacity:0.9;
  -webkit-transition-duration:0;
          transition-duration:0; }
  .bp3-card.bp3-interactive:active.bp3-dark,
  .bp3-dark .bp3-card.bp3-interactive:active{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-collapse{
  height:0;
  overflow-y:hidden;
  -webkit-transition:height 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:height 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-collapse .bp3-collapse-body{
    -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-collapse .bp3-collapse-body[aria-hidden="true"]{
      display:none; }

.bp3-context-menu .bp3-popover-target{
  display:block; }

.bp3-context-menu-popover-target{
  position:fixed; }

.bp3-divider{
  border-bottom:1px solid rgba(16, 22, 26, 0.15);
  border-right:1px solid rgba(16, 22, 26, 0.15);
  margin:5px; }
  .bp3-dark .bp3-divider{
    border-color:rgba(16, 22, 26, 0.4); }
.bp3-dialog-container{
  opacity:1;
  -webkit-transform:scale(1);
          transform:scale(1);
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  min-height:100%;
  pointer-events:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none;
  width:100%; }
  .bp3-dialog-container.bp3-overlay-enter > .bp3-dialog, .bp3-dialog-container.bp3-overlay-appear > .bp3-dialog{
    opacity:0;
    -webkit-transform:scale(0.5);
            transform:scale(0.5); }
  .bp3-dialog-container.bp3-overlay-enter-active > .bp3-dialog, .bp3-dialog-container.bp3-overlay-appear-active > .bp3-dialog{
    opacity:1;
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:opacity, transform;
    transition-property:opacity, transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-dialog-container.bp3-overlay-exit > .bp3-dialog{
    opacity:1;
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-dialog-container.bp3-overlay-exit-active > .bp3-dialog{
    opacity:0;
    -webkit-transform:scale(0.5);
            transform:scale(0.5);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:opacity, transform;
    transition-property:opacity, transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }

.bp3-dialog{
  background:#ebf1f5;
  border-radius:6px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:30px 0;
  padding-bottom:20px;
  pointer-events:all;
  -webkit-user-select:text;
     -moz-user-select:text;
      -ms-user-select:text;
          user-select:text;
  width:500px; }
  .bp3-dialog:focus{
    outline:0; }
  .bp3-dialog.bp3-dark,
  .bp3-dark .bp3-dialog{
    background:#293742;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }

.bp3-dialog-header{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background:#ffffff;
  border-radius:6px 6px 0 0;
  -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  min-height:40px;
  padding-left:20px;
  padding-right:5px; }
  .bp3-dialog-header .bp3-icon-large,
  .bp3-dialog-header .bp3-icon{
    color:#5c7080;
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    margin-right:10px; }
  .bp3-dialog-header .bp3-heading{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    line-height:inherit;
    margin:0; }
    .bp3-dialog-header .bp3-heading:last-child{
      margin-right:20px; }
  .bp3-dark .bp3-dialog-header{
    background:#30404d;
    -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:0 1px 0 rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-dialog-header .bp3-icon-large,
    .bp3-dark .bp3-dialog-header .bp3-icon{
      color:#a7b6c2; }

.bp3-dialog-body{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  line-height:18px;
  margin:20px; }

.bp3-dialog-footer{
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  margin:0 20px; }

.bp3-dialog-footer-actions{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:end;
      -ms-flex-pack:end;
          justify-content:flex-end; }
  .bp3-dialog-footer-actions .bp3-button{
    margin-left:10px; }
.bp3-drawer{
  background:#ffffff;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:0;
  padding:0; }
  .bp3-drawer:focus{
    outline:0; }
  .bp3-drawer.bp3-position-top{
    height:50%;
    left:0;
    right:0;
    top:0; }
    .bp3-drawer.bp3-position-top.bp3-overlay-enter, .bp3-drawer.bp3-position-top.bp3-overlay-appear{
      -webkit-transform:translateY(-100%);
              transform:translateY(-100%); }
    .bp3-drawer.bp3-position-top.bp3-overlay-enter-active, .bp3-drawer.bp3-position-top.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-top.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer.bp3-position-top.bp3-overlay-exit-active{
      -webkit-transform:translateY(-100%);
              transform:translateY(-100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-position-bottom{
    bottom:0;
    height:50%;
    left:0;
    right:0; }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-enter, .bp3-drawer.bp3-position-bottom.bp3-overlay-appear{
      -webkit-transform:translateY(100%);
              transform:translateY(100%); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-enter-active, .bp3-drawer.bp3-position-bottom.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-exit-active{
      -webkit-transform:translateY(100%);
              transform:translateY(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-position-left{
    bottom:0;
    left:0;
    top:0;
    width:50%; }
    .bp3-drawer.bp3-position-left.bp3-overlay-enter, .bp3-drawer.bp3-position-left.bp3-overlay-appear{
      -webkit-transform:translateX(-100%);
              transform:translateX(-100%); }
    .bp3-drawer.bp3-position-left.bp3-overlay-enter-active, .bp3-drawer.bp3-position-left.bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-left.bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer.bp3-position-left.bp3-overlay-exit-active{
      -webkit-transform:translateX(-100%);
              transform:translateX(-100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-position-right{
    bottom:0;
    right:0;
    top:0;
    width:50%; }
    .bp3-drawer.bp3-position-right.bp3-overlay-enter, .bp3-drawer.bp3-position-right.bp3-overlay-appear{
      -webkit-transform:translateX(100%);
              transform:translateX(100%); }
    .bp3-drawer.bp3-position-right.bp3-overlay-enter-active, .bp3-drawer.bp3-position-right.bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-right.bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer.bp3-position-right.bp3-overlay-exit-active{
      -webkit-transform:translateX(100%);
              transform:translateX(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
  .bp3-position-right):not(.bp3-vertical){
    bottom:0;
    right:0;
    top:0;
    width:50%; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-enter, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-appear{
      -webkit-transform:translateX(100%);
              transform:translateX(100%); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-enter-active, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-exit-active{
      -webkit-transform:translateX(100%);
              transform:translateX(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
  .bp3-position-right).bp3-vertical{
    bottom:0;
    height:50%;
    left:0;
    right:0; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-enter, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-appear{
      -webkit-transform:translateY(100%);
              transform:translateY(100%); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-enter-active, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-exit-active{
      -webkit-transform:translateY(100%);
              transform:translateY(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-dark,
  .bp3-dark .bp3-drawer{
    background:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }

.bp3-drawer-header{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  border-radius:0;
  -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  min-height:40px;
  padding:5px;
  padding-left:20px;
  position:relative; }
  .bp3-drawer-header .bp3-icon-large,
  .bp3-drawer-header .bp3-icon{
    color:#5c7080;
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    margin-right:10px; }
  .bp3-drawer-header .bp3-heading{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    line-height:inherit;
    margin:0; }
    .bp3-drawer-header .bp3-heading:last-child{
      margin-right:20px; }
  .bp3-dark .bp3-drawer-header{
    -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:0 1px 0 rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-drawer-header .bp3-icon-large,
    .bp3-dark .bp3-drawer-header .bp3-icon{
      color:#a7b6c2; }

.bp3-drawer-body{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  line-height:18px;
  overflow:auto; }

.bp3-drawer-footer{
  -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  padding:10px 20px;
  position:relative; }
  .bp3-dark .bp3-drawer-footer{
    -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.4); }
.bp3-editable-text{
  cursor:text;
  display:inline-block;
  max-width:100%;
  position:relative;
  vertical-align:top;
  white-space:nowrap; }
  .bp3-editable-text::before{
    bottom:-3px;
    left:-3px;
    position:absolute;
    right:-3px;
    top:-3px;
    border-radius:3px;
    content:"";
    -webkit-transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-editable-text:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-editable-text.bp3-editable-text-editing::before{
    background-color:#ffffff;
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-disabled::before{
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-editable-text.bp3-intent-primary .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-primary .bp3-editable-text-content{
    color:#137cbd; }
  .bp3-editable-text.bp3-intent-primary:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(19, 124, 189, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(19, 124, 189, 0.4); }
  .bp3-editable-text.bp3-intent-primary.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-success .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-success .bp3-editable-text-content{
    color:#0f9960; }
  .bp3-editable-text.bp3-intent-success:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px rgba(15, 153, 96, 0.4);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px rgba(15, 153, 96, 0.4); }
  .bp3-editable-text.bp3-intent-success.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-warning .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-warning .bp3-editable-text-content{
    color:#d9822b; }
  .bp3-editable-text.bp3-intent-warning:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px rgba(217, 130, 43, 0.4);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px rgba(217, 130, 43, 0.4); }
  .bp3-editable-text.bp3-intent-warning.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-danger .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-danger .bp3-editable-text-content{
    color:#db3737; }
  .bp3-editable-text.bp3-intent-danger:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px rgba(219, 55, 55, 0.4);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px rgba(219, 55, 55, 0.4); }
  .bp3-editable-text.bp3-intent-danger.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-editable-text:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(255, 255, 255, 0.15); }
  .bp3-dark .bp3-editable-text.bp3-editable-text-editing::before{
    background-color:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-disabled::before{
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-dark .bp3-editable-text.bp3-intent-primary .bp3-editable-text-content{
    color:#48aff0; }
  .bp3-dark .bp3-editable-text.bp3-intent-primary:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(72, 175, 240, 0), 0 0 0 0 rgba(72, 175, 240, 0), inset 0 0 0 1px rgba(72, 175, 240, 0.4);
            box-shadow:0 0 0 0 rgba(72, 175, 240, 0), 0 0 0 0 rgba(72, 175, 240, 0), inset 0 0 0 1px rgba(72, 175, 240, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-primary.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #48aff0, 0 0 0 3px rgba(72, 175, 240, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #48aff0, 0 0 0 3px rgba(72, 175, 240, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-success .bp3-editable-text-content{
    color:#3dcc91; }
  .bp3-dark .bp3-editable-text.bp3-intent-success:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(61, 204, 145, 0), 0 0 0 0 rgba(61, 204, 145, 0), inset 0 0 0 1px rgba(61, 204, 145, 0.4);
            box-shadow:0 0 0 0 rgba(61, 204, 145, 0), 0 0 0 0 rgba(61, 204, 145, 0), inset 0 0 0 1px rgba(61, 204, 145, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-success.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #3dcc91, 0 0 0 3px rgba(61, 204, 145, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #3dcc91, 0 0 0 3px rgba(61, 204, 145, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-warning .bp3-editable-text-content{
    color:#ffb366; }
  .bp3-dark .bp3-editable-text.bp3-intent-warning:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(255, 179, 102, 0), 0 0 0 0 rgba(255, 179, 102, 0), inset 0 0 0 1px rgba(255, 179, 102, 0.4);
            box-shadow:0 0 0 0 rgba(255, 179, 102, 0), 0 0 0 0 rgba(255, 179, 102, 0), inset 0 0 0 1px rgba(255, 179, 102, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-warning.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #ffb366, 0 0 0 3px rgba(255, 179, 102, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #ffb366, 0 0 0 3px rgba(255, 179, 102, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-danger .bp3-editable-text-content{
    color:#ff7373; }
  .bp3-dark .bp3-editable-text.bp3-intent-danger:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(255, 115, 115, 0), 0 0 0 0 rgba(255, 115, 115, 0), inset 0 0 0 1px rgba(255, 115, 115, 0.4);
            box-shadow:0 0 0 0 rgba(255, 115, 115, 0), 0 0 0 0 rgba(255, 115, 115, 0), inset 0 0 0 1px rgba(255, 115, 115, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-danger.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #ff7373, 0 0 0 3px rgba(255, 115, 115, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #ff7373, 0 0 0 3px rgba(255, 115, 115, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-editable-text-input,
.bp3-editable-text-content{
  color:inherit;
  display:inherit;
  font:inherit;
  letter-spacing:inherit;
  max-width:inherit;
  min-width:inherit;
  position:relative;
  resize:none;
  text-transform:inherit;
  vertical-align:top; }

.bp3-editable-text-input{
  background:none;
  border:none;
  -webkit-box-shadow:none;
          box-shadow:none;
  padding:0;
  white-space:pre-wrap;
  width:100%; }
  .bp3-editable-text-input::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input:focus{
    outline:none; }
  .bp3-editable-text-input::-ms-clear{
    display:none; }

.bp3-editable-text-content{
  overflow:hidden;
  padding-right:2px;
  text-overflow:ellipsis;
  white-space:pre; }
  .bp3-editable-text-editing > .bp3-editable-text-content{
    left:0;
    position:absolute;
    visibility:hidden; }
  .bp3-editable-text-placeholder > .bp3-editable-text-content{
    color:rgba(92, 112, 128, 0.6); }
    .bp3-dark .bp3-editable-text-placeholder > .bp3-editable-text-content{
      color:rgba(167, 182, 194, 0.6); }

.bp3-editable-text.bp3-multiline{
  display:block; }
  .bp3-editable-text.bp3-multiline .bp3-editable-text-content{
    overflow:auto;
    white-space:pre-wrap;
    word-wrap:break-word; }
.bp3-divider{
  border-bottom:1px solid rgba(16, 22, 26, 0.15);
  border-right:1px solid rgba(16, 22, 26, 0.15);
  margin:5px; }
  .bp3-dark .bp3-divider{
    border-color:rgba(16, 22, 26, 0.4); }
.bp3-control-group{
  -webkit-transform:translateZ(0);
          transform:translateZ(0);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:stretch;
      -ms-flex-align:stretch;
          align-items:stretch; }
  .bp3-control-group > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-control-group > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-control-group .bp3-button,
  .bp3-control-group .bp3-html-select,
  .bp3-control-group .bp3-input,
  .bp3-control-group .bp3-select{
    position:relative; }
  .bp3-control-group .bp3-input{
    border-radius:inherit;
    z-index:2; }
    .bp3-control-group .bp3-input:focus{
      border-radius:3px;
      z-index:14; }
    .bp3-control-group .bp3-input[class*="bp3-intent"]{
      z-index:13; }
      .bp3-control-group .bp3-input[class*="bp3-intent"]:focus{
        z-index:15; }
    .bp3-control-group .bp3-input[readonly], .bp3-control-group .bp3-input:disabled, .bp3-control-group .bp3-input.bp3-disabled{
      z-index:1; }
  .bp3-control-group .bp3-input-group[class*="bp3-intent"] .bp3-input{
    z-index:13; }
    .bp3-control-group .bp3-input-group[class*="bp3-intent"] .bp3-input:focus{
      z-index:15; }
  .bp3-control-group .bp3-button,
  .bp3-control-group .bp3-html-select select,
  .bp3-control-group .bp3-select select{
    -webkit-transform:translateZ(0);
            transform:translateZ(0);
    border-radius:inherit;
    z-index:4; }
    .bp3-control-group .bp3-button:focus,
    .bp3-control-group .bp3-html-select select:focus,
    .bp3-control-group .bp3-select select:focus{
      z-index:5; }
    .bp3-control-group .bp3-button:hover,
    .bp3-control-group .bp3-html-select select:hover,
    .bp3-control-group .bp3-select select:hover{
      z-index:6; }
    .bp3-control-group .bp3-button:active,
    .bp3-control-group .bp3-html-select select:active,
    .bp3-control-group .bp3-select select:active{
      z-index:7; }
    .bp3-control-group .bp3-button[readonly], .bp3-control-group .bp3-button:disabled, .bp3-control-group .bp3-button.bp3-disabled,
    .bp3-control-group .bp3-html-select select[readonly],
    .bp3-control-group .bp3-html-select select:disabled,
    .bp3-control-group .bp3-html-select select.bp3-disabled,
    .bp3-control-group .bp3-select select[readonly],
    .bp3-control-group .bp3-select select:disabled,
    .bp3-control-group .bp3-select select.bp3-disabled{
      z-index:3; }
    .bp3-control-group .bp3-button[class*="bp3-intent"],
    .bp3-control-group .bp3-html-select select[class*="bp3-intent"],
    .bp3-control-group .bp3-select select[class*="bp3-intent"]{
      z-index:9; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:focus,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:focus,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:focus{
        z-index:10; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:hover,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:hover,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:hover{
        z-index:11; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:active,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:active,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:active{
        z-index:12; }
      .bp3-control-group .bp3-button[class*="bp3-intent"][readonly], .bp3-control-group .bp3-button[class*="bp3-intent"]:disabled, .bp3-control-group .bp3-button[class*="bp3-intent"].bp3-disabled,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"][readonly],
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:disabled,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"].bp3-disabled,
      .bp3-control-group .bp3-select select[class*="bp3-intent"][readonly],
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:disabled,
      .bp3-control-group .bp3-select select[class*="bp3-intent"].bp3-disabled{
        z-index:8; }
  .bp3-control-group .bp3-input-group > .bp3-icon,
  .bp3-control-group .bp3-input-group > .bp3-button,
  .bp3-control-group .bp3-input-group > .bp3-input-action{
    z-index:16; }
  .bp3-control-group .bp3-select::after,
  .bp3-control-group .bp3-html-select::after,
  .bp3-control-group .bp3-select > .bp3-icon,
  .bp3-control-group .bp3-html-select > .bp3-icon{
    z-index:17; }
  .bp3-control-group .bp3-select:focus-within{
    z-index:5; }
  .bp3-control-group:not(.bp3-vertical) > *:not(.bp3-divider){
    margin-right:-1px; }
  .bp3-control-group:not(.bp3-vertical) > .bp3-divider:not(:first-child){
    margin-left:6px; }
  .bp3-dark .bp3-control-group:not(.bp3-vertical) > *:not(.bp3-divider){
    margin-right:0; }
  .bp3-dark .bp3-control-group:not(.bp3-vertical) > .bp3-button + .bp3-button{
    margin-left:1px; }
  .bp3-control-group .bp3-popover-wrapper,
  .bp3-control-group .bp3-popover-target{
    border-radius:inherit; }
  .bp3-control-group > :first-child{
    border-radius:3px 0 0 3px; }
  .bp3-control-group > :last-child{
    border-radius:0 3px 3px 0;
    margin-right:0; }
  .bp3-control-group > :only-child{
    border-radius:3px;
    margin-right:0; }
  .bp3-control-group .bp3-input-group .bp3-button{
    border-radius:3px; }
  .bp3-control-group .bp3-numeric-input:not(:first-child) .bp3-input-group{
    border-bottom-left-radius:0;
    border-top-left-radius:0; }
  .bp3-control-group.bp3-fill{
    width:100%; }
  .bp3-control-group > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-control-group.bp3-fill > *:not(.bp3-fixed){
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-control-group.bp3-vertical{
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column; }
    .bp3-control-group.bp3-vertical > *{
      margin-top:-1px; }
    .bp3-control-group.bp3-vertical > :first-child{
      border-radius:3px 3px 0 0;
      margin-top:0; }
    .bp3-control-group.bp3-vertical > :last-child{
      border-radius:0 0 3px 3px; }
.bp3-control{
  cursor:pointer;
  display:block;
  margin-bottom:10px;
  position:relative;
  text-transform:none; }
  .bp3-control input:checked ~ .bp3-control-indicator{
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
  .bp3-control:hover input:checked ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
  .bp3-control input:not(:disabled):active:checked ~ .bp3-control-indicator{
    background:#0e5a8a;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-control input:disabled:checked ~ .bp3-control-indicator{
    background:rgba(19, 124, 189, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-dark .bp3-control input:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control:hover input:checked ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control input:not(:disabled):active:checked ~ .bp3-control-indicator{
    background-color:#0e5a8a;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-control input:disabled:checked ~ .bp3-control-indicator{
    background:rgba(14, 90, 138, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-control:not(.bp3-align-right){
    padding-left:26px; }
    .bp3-control:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-26px; }
  .bp3-control.bp3-align-right{
    padding-right:26px; }
    .bp3-control.bp3-align-right .bp3-control-indicator{
      margin-right:-26px; }
  .bp3-control.bp3-disabled{
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  .bp3-control.bp3-inline{
    display:inline-block;
    margin-right:20px; }
  .bp3-control input{
    left:0;
    opacity:0;
    position:absolute;
    top:0;
    z-index:-1; }
  .bp3-control .bp3-control-indicator{
    background-clip:padding-box;
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    border:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    cursor:pointer;
    display:inline-block;
    font-size:16px;
    height:1em;
    margin-right:10px;
    margin-top:-3px;
    position:relative;
    -webkit-user-select:none;
       -moz-user-select:none;
        -ms-user-select:none;
            user-select:none;
    vertical-align:middle;
    width:1em; }
    .bp3-control .bp3-control-indicator::before{
      content:"";
      display:block;
      height:1em;
      width:1em; }
  .bp3-control:hover .bp3-control-indicator{
    background-color:#ebf1f5; }
  .bp3-control input:not(:disabled):active ~ .bp3-control-indicator{
    background:#d8e1e8;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-control input:disabled ~ .bp3-control-indicator{
    background:rgba(206, 217, 224, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none;
    cursor:not-allowed; }
  .bp3-control input:focus ~ .bp3-control-indicator{
    outline:rgba(19, 124, 189, 0.6) auto 2px;
    outline-offset:2px;
    -moz-outline-radius:6px; }
  .bp3-control.bp3-align-right .bp3-control-indicator{
    float:right;
    margin-left:10px;
    margin-top:1px; }
  .bp3-control.bp3-large{
    font-size:16px; }
    .bp3-control.bp3-large:not(.bp3-align-right){
      padding-left:30px; }
      .bp3-control.bp3-large:not(.bp3-align-right) .bp3-control-indicator{
        margin-left:-30px; }
    .bp3-control.bp3-large.bp3-align-right{
      padding-right:30px; }
      .bp3-control.bp3-large.bp3-align-right .bp3-control-indicator{
        margin-right:-30px; }
    .bp3-control.bp3-large .bp3-control-indicator{
      font-size:20px; }
    .bp3-control.bp3-large.bp3-align-right .bp3-control-indicator{
      margin-top:0; }
  .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator{
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
  .bp3-control.bp3-checkbox:hover input:indeterminate ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
  .bp3-control.bp3-checkbox input:not(:disabled):active:indeterminate ~ .bp3-control-indicator{
    background:#0e5a8a;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
    background:rgba(19, 124, 189, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-dark .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-checkbox:hover input:indeterminate ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-checkbox input:not(:disabled):active:indeterminate ~ .bp3-control-indicator{
    background-color:#0e5a8a;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
    background:rgba(14, 90, 138, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-control.bp3-checkbox .bp3-control-indicator{
    border-radius:3px; }
  .bp3-control.bp3-checkbox input:checked ~ .bp3-control-indicator::before{
    background-image:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M12 5c-.28 0-.53.11-.71.29L7 9.59l-2.29-2.3a1.003 1.003 0 00-1.42 1.42l3 3c.18.18.43.29.71.29s.53-.11.71-.29l5-5A1.003 1.003 0 0012 5z' fill='white'/%3e%3c/svg%3e"); }
  .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator::before{
    background-image:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M11 7H5c-.55 0-1 .45-1 1s.45 1 1 1h6c.55 0 1-.45 1-1s-.45-1-1-1z' fill='white'/%3e%3c/svg%3e"); }
  .bp3-control.bp3-radio .bp3-control-indicator{
    border-radius:50%; }
  .bp3-control.bp3-radio input:checked ~ .bp3-control-indicator::before{
    background-image:radial-gradient(#ffffff, #ffffff 28%, transparent 32%); }
  .bp3-control.bp3-radio input:checked:disabled ~ .bp3-control-indicator::before{
    opacity:0.5; }
  .bp3-control.bp3-radio input:focus ~ .bp3-control-indicator{
    -moz-outline-radius:16px; }
  .bp3-control.bp3-switch input ~ .bp3-control-indicator{
    background:rgba(167, 182, 194, 0.5); }
  .bp3-control.bp3-switch:hover input ~ .bp3-control-indicator{
    background:rgba(115, 134, 148, 0.5); }
  .bp3-control.bp3-switch input:not(:disabled):active ~ .bp3-control-indicator{
    background:rgba(92, 112, 128, 0.5); }
  .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator{
    background:rgba(206, 217, 224, 0.5); }
    .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator::before{
      background:rgba(255, 255, 255, 0.8); }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator{
    background:#137cbd; }
  .bp3-control.bp3-switch:hover input:checked ~ .bp3-control-indicator{
    background:#106ba3; }
  .bp3-control.bp3-switch input:checked:not(:disabled):active ~ .bp3-control-indicator{
    background:#0e5a8a; }
  .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator{
    background:rgba(19, 124, 189, 0.5); }
    .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator::before{
      background:rgba(255, 255, 255, 0.8); }
  .bp3-control.bp3-switch:not(.bp3-align-right){
    padding-left:38px; }
    .bp3-control.bp3-switch:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-38px; }
  .bp3-control.bp3-switch.bp3-align-right{
    padding-right:38px; }
    .bp3-control.bp3-switch.bp3-align-right .bp3-control-indicator{
      margin-right:-38px; }
  .bp3-control.bp3-switch .bp3-control-indicator{
    border:none;
    border-radius:1.75em;
    -webkit-box-shadow:none !important;
            box-shadow:none !important;
    min-width:1.75em;
    -webkit-transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    width:auto; }
    .bp3-control.bp3-switch .bp3-control-indicator::before{
      background:#ffffff;
      border-radius:50%;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
      height:calc(1em - 4px);
      left:0;
      margin:2px;
      position:absolute;
      -webkit-transition:left 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
      transition:left 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
      width:calc(1em - 4px); }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator::before{
    left:calc(100% - 1em); }
  .bp3-control.bp3-switch.bp3-large:not(.bp3-align-right){
    padding-left:45px; }
    .bp3-control.bp3-switch.bp3-large:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-45px; }
  .bp3-control.bp3-switch.bp3-large.bp3-align-right{
    padding-right:45px; }
    .bp3-control.bp3-switch.bp3-large.bp3-align-right .bp3-control-indicator{
      margin-right:-45px; }
  .bp3-dark .bp3-control.bp3-switch input ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.5); }
  .bp3-dark .bp3-control.bp3-switch:hover input ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.7); }
  .bp3-dark .bp3-control.bp3-switch input:not(:disabled):active ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.9); }
  .bp3-dark .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator{
    background:rgba(57, 75, 89, 0.5); }
    .bp3-dark .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator::before{
      background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator{
    background:#137cbd; }
  .bp3-dark .bp3-control.bp3-switch:hover input:checked ~ .bp3-control-indicator{
    background:#106ba3; }
  .bp3-dark .bp3-control.bp3-switch input:checked:not(:disabled):active ~ .bp3-control-indicator{
    background:#0e5a8a; }
  .bp3-dark .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator{
    background:rgba(14, 90, 138, 0.5); }
    .bp3-dark .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator::before{
      background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch .bp3-control-indicator::before{
    background:#394b59;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator::before{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-control.bp3-switch .bp3-switch-inner-text{
    font-size:0.7em;
    text-align:center; }
  .bp3-control.bp3-switch .bp3-control-indicator-child:first-child{
    line-height:0;
    margin-left:0.5em;
    margin-right:1.2em;
    visibility:hidden; }
  .bp3-control.bp3-switch .bp3-control-indicator-child:last-child{
    line-height:1em;
    margin-left:1.2em;
    margin-right:0.5em;
    visibility:visible; }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator .bp3-control-indicator-child:first-child{
    line-height:1em;
    visibility:visible; }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator .bp3-control-indicator-child:last-child{
    line-height:0;
    visibility:hidden; }
  .bp3-dark .bp3-control{
    color:#f5f8fa; }
    .bp3-dark .bp3-control.bp3-disabled{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-control .bp3-control-indicator{
      background-color:#394b59;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-control:hover .bp3-control-indicator{
      background-color:#30404d; }
    .bp3-dark .bp3-control input:not(:disabled):active ~ .bp3-control-indicator{
      background:#202b33;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-control input:disabled ~ .bp3-control-indicator{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      cursor:not-allowed; }
    .bp3-dark .bp3-control.bp3-checkbox input:disabled:checked ~ .bp3-control-indicator, .bp3-dark .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
      color:rgba(167, 182, 194, 0.6); }
.bp3-file-input{
  cursor:pointer;
  display:inline-block;
  height:30px;
  position:relative; }
  .bp3-file-input input{
    margin:0;
    min-width:200px;
    opacity:0; }
    .bp3-file-input input:disabled + .bp3-file-upload-input,
    .bp3-file-input input.bp3-disabled + .bp3-file-upload-input{
      background:rgba(206, 217, 224, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      resize:none; }
      .bp3-file-input input:disabled + .bp3-file-upload-input::after,
      .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after{
        background-color:rgba(206, 217, 224, 0.5);
        background-image:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(92, 112, 128, 0.6);
        cursor:not-allowed;
        outline:none; }
        .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active, .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active:hover,
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active,
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active:hover{
          background:rgba(206, 217, 224, 0.7); }
      .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input, .bp3-dark
      .bp3-file-input input.bp3-disabled + .bp3-file-upload-input{
        background:rgba(57, 75, 89, 0.5);
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input::after, .bp3-dark
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after{
          background-color:rgba(57, 75, 89, 0.5);
          background-image:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:rgba(167, 182, 194, 0.6); }
          .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active, .bp3-dark
          .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active{
            background:rgba(57, 75, 89, 0.7); }
  .bp3-file-input.bp3-file-input-has-selection .bp3-file-upload-input{
    color:#182026; }
  .bp3-dark .bp3-file-input.bp3-file-input-has-selection .bp3-file-upload-input{
    color:#f5f8fa; }
  .bp3-file-input.bp3-fill{
    width:100%; }
  .bp3-file-input.bp3-large,
  .bp3-large .bp3-file-input{
    height:40px; }
  .bp3-file-input .bp3-file-upload-input-custom-text::after{
    content:attr(bp3-button-text); }

.bp3-file-upload-input{
  -webkit-appearance:none;
     -moz-appearance:none;
          appearance:none;
  background:#ffffff;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
  color:#182026;
  font-size:14px;
  font-weight:400;
  height:30px;
  line-height:30px;
  outline:none;
  padding:0 10px;
  -webkit-transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  vertical-align:middle;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  color:rgba(92, 112, 128, 0.6);
  left:0;
  padding-right:80px;
  position:absolute;
  right:0;
  top:0;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-file-upload-input::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input:focus, .bp3-file-upload-input.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-file-upload-input[type="search"], .bp3-file-upload-input.bp3-round{
    border-radius:30px;
    -webkit-box-sizing:border-box;
            box-sizing:border-box;
    padding-left:10px; }
  .bp3-file-upload-input[readonly]{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-file-upload-input:disabled, .bp3-file-upload-input.bp3-disabled{
    background:rgba(206, 217, 224, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    resize:none; }
  .bp3-file-upload-input::after{
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    color:#182026;
    min-height:24px;
    min-width:24px;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    border-radius:3px;
    content:"Browse";
    line-height:24px;
    margin:3px;
    position:absolute;
    right:0;
    text-align:center;
    top:0;
    width:70px; }
    .bp3-file-upload-input::after:hover{
      background-clip:padding-box;
      background-color:#ebf1f5;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
    .bp3-file-upload-input::after:active, .bp3-file-upload-input::after.bp3-active{
      background-color:#d8e1e8;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-file-upload-input::after:disabled, .bp3-file-upload-input::after.bp3-disabled{
      background-color:rgba(206, 217, 224, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      outline:none; }
      .bp3-file-upload-input::after:disabled.bp3-active, .bp3-file-upload-input::after:disabled.bp3-active:hover, .bp3-file-upload-input::after.bp3-disabled.bp3-active, .bp3-file-upload-input::after.bp3-disabled.bp3-active:hover{
        background:rgba(206, 217, 224, 0.7); }
  .bp3-file-upload-input:hover::after{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
  .bp3-file-upload-input:active::after{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-large .bp3-file-upload-input{
    font-size:16px;
    height:40px;
    line-height:40px;
    padding-right:95px; }
    .bp3-large .bp3-file-upload-input[type="search"], .bp3-large .bp3-file-upload-input.bp3-round{
      padding:0 15px; }
    .bp3-large .bp3-file-upload-input::after{
      min-height:30px;
      min-width:30px;
      line-height:30px;
      margin:5px;
      width:85px; }
  .bp3-dark .bp3-file-upload-input{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa;
    color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input:disabled, .bp3-dark .bp3-file-upload-input.bp3-disabled{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::after{
      background-color:#394b59;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      color:#f5f8fa; }
      .bp3-dark .bp3-file-upload-input::after:hover, .bp3-dark .bp3-file-upload-input::after:active, .bp3-dark .bp3-file-upload-input::after.bp3-active{
        color:#f5f8fa; }
      .bp3-dark .bp3-file-upload-input::after:hover{
        background-color:#30404d;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-file-upload-input::after:active, .bp3-dark .bp3-file-upload-input::after.bp3-active{
        background-color:#202b33;
        background-image:none;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
      .bp3-dark .bp3-file-upload-input::after:disabled, .bp3-dark .bp3-file-upload-input::after.bp3-disabled{
        background-color:rgba(57, 75, 89, 0.5);
        background-image:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-file-upload-input::after:disabled.bp3-active, .bp3-dark .bp3-file-upload-input::after.bp3-disabled.bp3-active{
          background:rgba(57, 75, 89, 0.7); }
      .bp3-dark .bp3-file-upload-input::after .bp3-button-spinner .bp3-spinner-head{
        background:rgba(16, 22, 26, 0.5);
        stroke:#8a9ba8; }
    .bp3-dark .bp3-file-upload-input:hover::after{
      background-color:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input:active::after{
      background-color:#202b33;
      background-image:none;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
.bp3-file-upload-input::after{
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
.bp3-form-group{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:0 0 15px; }
  .bp3-form-group label.bp3-label{
    margin-bottom:5px; }
  .bp3-form-group .bp3-control{
    margin-top:7px; }
  .bp3-form-group .bp3-form-helper-text{
    color:#5c7080;
    font-size:12px;
    margin-top:5px; }
  .bp3-form-group.bp3-intent-primary .bp3-form-helper-text{
    color:#106ba3; }
  .bp3-form-group.bp3-intent-success .bp3-form-helper-text{
    color:#0d8050; }
  .bp3-form-group.bp3-intent-warning .bp3-form-helper-text{
    color:#bf7326; }
  .bp3-form-group.bp3-intent-danger .bp3-form-helper-text{
    color:#c23030; }
  .bp3-form-group.bp3-inline{
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row; }
    .bp3-form-group.bp3-inline.bp3-large label.bp3-label{
      line-height:40px;
      margin:0 10px 0 0; }
    .bp3-form-group.bp3-inline label.bp3-label{
      line-height:30px;
      margin:0 10px 0 0; }
  .bp3-form-group.bp3-disabled .bp3-label,
  .bp3-form-group.bp3-disabled .bp3-text-muted,
  .bp3-form-group.bp3-disabled .bp3-form-helper-text{
    color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-dark .bp3-form-group.bp3-intent-primary .bp3-form-helper-text{
    color:#48aff0; }
  .bp3-dark .bp3-form-group.bp3-intent-success .bp3-form-helper-text{
    color:#3dcc91; }
  .bp3-dark .bp3-form-group.bp3-intent-warning .bp3-form-helper-text{
    color:#ffb366; }
  .bp3-dark .bp3-form-group.bp3-intent-danger .bp3-form-helper-text{
    color:#ff7373; }
  .bp3-dark .bp3-form-group .bp3-form-helper-text{
    color:#a7b6c2; }
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-label,
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-text-muted,
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-form-helper-text{
    color:rgba(167, 182, 194, 0.6) !important; }
.bp3-input-group{
  display:block;
  position:relative; }
  .bp3-input-group .bp3-input{
    position:relative;
    width:100%; }
    .bp3-input-group .bp3-input:not(:first-child){
      padding-left:30px; }
    .bp3-input-group .bp3-input:not(:last-child){
      padding-right:30px; }
  .bp3-input-group .bp3-input-action,
  .bp3-input-group > .bp3-input-left-container,
  .bp3-input-group > .bp3-button,
  .bp3-input-group > .bp3-icon{
    position:absolute;
    top:0; }
    .bp3-input-group .bp3-input-action:first-child,
    .bp3-input-group > .bp3-input-left-container:first-child,
    .bp3-input-group > .bp3-button:first-child,
    .bp3-input-group > .bp3-icon:first-child{
      left:0; }
    .bp3-input-group .bp3-input-action:last-child,
    .bp3-input-group > .bp3-input-left-container:last-child,
    .bp3-input-group > .bp3-button:last-child,
    .bp3-input-group > .bp3-icon:last-child{
      right:0; }
  .bp3-input-group .bp3-button{
    min-height:24px;
    min-width:24px;
    margin:3px;
    padding:0 7px; }
    .bp3-input-group .bp3-button:empty{
      padding:0; }
  .bp3-input-group > .bp3-input-left-container,
  .bp3-input-group > .bp3-icon{
    z-index:1; }
  .bp3-input-group > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group > .bp3-icon{
    color:#5c7080; }
    .bp3-input-group > .bp3-input-left-container > .bp3-icon:empty,
    .bp3-input-group > .bp3-icon:empty{
      font-family:"Icons16", sans-serif;
      font-size:16px;
      font-style:normal;
      font-weight:400;
      line-height:1;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased; }
  .bp3-input-group > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group > .bp3-icon,
  .bp3-input-group .bp3-input-action > .bp3-spinner{
    margin:7px; }
  .bp3-input-group .bp3-tag{
    margin:5px; }
  .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus),
  .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus){
    color:#5c7080; }
    .bp3-dark .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus), .bp3-dark
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus){
      color:#a7b6c2; }
    .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-standard, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-large,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-standard,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-large{
      color:#5c7080; }
  .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled,
  .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled{
    color:rgba(92, 112, 128, 0.6) !important; }
    .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon-standard, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon-large,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon-standard,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon-large{
      color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-input-group.bp3-disabled{
    cursor:not-allowed; }
    .bp3-input-group.bp3-disabled .bp3-icon{
      color:rgba(92, 112, 128, 0.6); }
  .bp3-input-group.bp3-large .bp3-button{
    min-height:30px;
    min-width:30px;
    margin:5px; }
  .bp3-input-group.bp3-large > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group.bp3-large > .bp3-icon,
  .bp3-input-group.bp3-large .bp3-input-action > .bp3-spinner{
    margin:12px; }
  .bp3-input-group.bp3-large .bp3-input{
    font-size:16px;
    height:40px;
    line-height:40px; }
    .bp3-input-group.bp3-large .bp3-input[type="search"], .bp3-input-group.bp3-large .bp3-input.bp3-round{
      padding:0 15px; }
    .bp3-input-group.bp3-large .bp3-input:not(:first-child){
      padding-left:40px; }
    .bp3-input-group.bp3-large .bp3-input:not(:last-child){
      padding-right:40px; }
  .bp3-input-group.bp3-small .bp3-button{
    min-height:20px;
    min-width:20px;
    margin:2px; }
  .bp3-input-group.bp3-small .bp3-tag{
    min-height:20px;
    min-width:20px;
    margin:2px; }
  .bp3-input-group.bp3-small > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group.bp3-small > .bp3-icon,
  .bp3-input-group.bp3-small .bp3-input-action > .bp3-spinner{
    margin:4px; }
  .bp3-input-group.bp3-small .bp3-input{
    font-size:12px;
    height:24px;
    line-height:24px;
    padding-left:8px;
    padding-right:8px; }
    .bp3-input-group.bp3-small .bp3-input[type="search"], .bp3-input-group.bp3-small .bp3-input.bp3-round{
      padding:0 12px; }
    .bp3-input-group.bp3-small .bp3-input:not(:first-child){
      padding-left:24px; }
    .bp3-input-group.bp3-small .bp3-input:not(:last-child){
      padding-right:24px; }
  .bp3-input-group.bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    width:100%; }
  .bp3-input-group.bp3-round .bp3-button,
  .bp3-input-group.bp3-round .bp3-input,
  .bp3-input-group.bp3-round .bp3-tag{
    border-radius:30px; }
  .bp3-dark .bp3-input-group .bp3-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-input-group.bp3-disabled .bp3-icon{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-input-group.bp3-intent-primary .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-primary .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-primary .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #137cbd;
              box-shadow:inset 0 0 0 1px #137cbd; }
    .bp3-input-group.bp3-intent-primary .bp3-input:disabled, .bp3-input-group.bp3-intent-primary .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-primary > .bp3-icon{
    color:#106ba3; }
    .bp3-dark .bp3-input-group.bp3-intent-primary > .bp3-icon{
      color:#48aff0; }
  .bp3-input-group.bp3-intent-success .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-success .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-success .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #0f9960;
              box-shadow:inset 0 0 0 1px #0f9960; }
    .bp3-input-group.bp3-intent-success .bp3-input:disabled, .bp3-input-group.bp3-intent-success .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-success > .bp3-icon{
    color:#0d8050; }
    .bp3-dark .bp3-input-group.bp3-intent-success > .bp3-icon{
      color:#3dcc91; }
  .bp3-input-group.bp3-intent-warning .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-warning .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-warning .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #d9822b;
              box-shadow:inset 0 0 0 1px #d9822b; }
    .bp3-input-group.bp3-intent-warning .bp3-input:disabled, .bp3-input-group.bp3-intent-warning .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-warning > .bp3-icon{
    color:#bf7326; }
    .bp3-dark .bp3-input-group.bp3-intent-warning > .bp3-icon{
      color:#ffb366; }
  .bp3-input-group.bp3-intent-danger .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-danger .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-danger .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #db3737;
              box-shadow:inset 0 0 0 1px #db3737; }
    .bp3-input-group.bp3-intent-danger .bp3-input:disabled, .bp3-input-group.bp3-intent-danger .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-danger > .bp3-icon{
    color:#c23030; }
    .bp3-dark .bp3-input-group.bp3-intent-danger > .bp3-icon{
      color:#ff7373; }
.bp3-input{
  -webkit-appearance:none;
     -moz-appearance:none;
          appearance:none;
  background:#ffffff;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
  color:#182026;
  font-size:14px;
  font-weight:400;
  height:30px;
  line-height:30px;
  outline:none;
  padding:0 10px;
  -webkit-transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  vertical-align:middle; }
  .bp3-input::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input:focus, .bp3-input.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-input[type="search"], .bp3-input.bp3-round{
    border-radius:30px;
    -webkit-box-sizing:border-box;
            box-sizing:border-box;
    padding-left:10px; }
  .bp3-input[readonly]{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-input:disabled, .bp3-input.bp3-disabled{
    background:rgba(206, 217, 224, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    resize:none; }
  .bp3-input.bp3-large{
    font-size:16px;
    height:40px;
    line-height:40px; }
    .bp3-input.bp3-large[type="search"], .bp3-input.bp3-large.bp3-round{
      padding:0 15px; }
  .bp3-input.bp3-small{
    font-size:12px;
    height:24px;
    line-height:24px;
    padding-left:8px;
    padding-right:8px; }
    .bp3-input.bp3-small[type="search"], .bp3-input.bp3-small.bp3-round{
      padding:0 12px; }
  .bp3-input.bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    width:100%; }
  .bp3-dark .bp3-input{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark .bp3-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-input:disabled, .bp3-dark .bp3-input.bp3-disabled{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
  .bp3-input.bp3-intent-primary{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-primary:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-primary[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #137cbd;
              box-shadow:inset 0 0 0 1px #137cbd; }
    .bp3-input.bp3-intent-primary:disabled, .bp3-input.bp3-intent-primary.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-primary:focus{
        -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-primary[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #137cbd;
                box-shadow:inset 0 0 0 1px #137cbd; }
      .bp3-dark .bp3-input.bp3-intent-primary:disabled, .bp3-dark .bp3-input.bp3-intent-primary.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-success{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-success:focus{
      -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-success[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #0f9960;
              box-shadow:inset 0 0 0 1px #0f9960; }
    .bp3-input.bp3-intent-success:disabled, .bp3-input.bp3-intent-success.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-success{
      -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-success:focus{
        -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #0f9960, 0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-success[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #0f9960;
                box-shadow:inset 0 0 0 1px #0f9960; }
      .bp3-dark .bp3-input.bp3-intent-success:disabled, .bp3-dark .bp3-input.bp3-intent-success.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-warning{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-warning:focus{
      -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-warning[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #d9822b;
              box-shadow:inset 0 0 0 1px #d9822b; }
    .bp3-input.bp3-intent-warning:disabled, .bp3-input.bp3-intent-warning.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-warning:focus{
        -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #d9822b, 0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-warning[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #d9822b;
                box-shadow:inset 0 0 0 1px #d9822b; }
      .bp3-dark .bp3-input.bp3-intent-warning:disabled, .bp3-dark .bp3-input.bp3-intent-warning.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-danger{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-danger:focus{
      -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-danger[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #db3737;
              box-shadow:inset 0 0 0 1px #db3737; }
    .bp3-input.bp3-intent-danger:disabled, .bp3-input.bp3-intent-danger.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-danger:focus{
        -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #db3737, 0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-danger[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #db3737;
                box-shadow:inset 0 0 0 1px #db3737; }
      .bp3-dark .bp3-input.bp3-intent-danger:disabled, .bp3-dark .bp3-input.bp3-intent-danger.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input::-ms-clear{
    display:none; }
textarea.bp3-input{
  max-width:100%;
  padding:10px; }
  textarea.bp3-input, textarea.bp3-input.bp3-large, textarea.bp3-input.bp3-small{
    height:auto;
    line-height:inherit; }
  textarea.bp3-input.bp3-small{
    padding:8px; }
  .bp3-dark textarea.bp3-input{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark textarea.bp3-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark textarea.bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark textarea.bp3-input:disabled, .bp3-dark textarea.bp3-input.bp3-disabled{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
label.bp3-label{
  display:block;
  margin-bottom:15px;
  margin-top:0; }
  label.bp3-label .bp3-html-select,
  label.bp3-label .bp3-input,
  label.bp3-label .bp3-select,
  label.bp3-label .bp3-slider,
  label.bp3-label .bp3-popover-wrapper{
    display:block;
    margin-top:5px;
    text-transform:none; }
  label.bp3-label .bp3-button-group{
    margin-top:5px; }
  label.bp3-label .bp3-select select,
  label.bp3-label .bp3-html-select select{
    font-weight:400;
    vertical-align:top;
    width:100%; }
  label.bp3-label.bp3-disabled,
  label.bp3-label.bp3-disabled .bp3-text-muted{
    color:rgba(92, 112, 128, 0.6); }
  label.bp3-label.bp3-inline{
    line-height:30px; }
    label.bp3-label.bp3-inline .bp3-html-select,
    label.bp3-label.bp3-inline .bp3-input,
    label.bp3-label.bp3-inline .bp3-input-group,
    label.bp3-label.bp3-inline .bp3-select,
    label.bp3-label.bp3-inline .bp3-popover-wrapper{
      display:inline-block;
      margin:0 0 0 5px;
      vertical-align:top; }
    label.bp3-label.bp3-inline .bp3-button-group{
      margin:0 0 0 5px; }
    label.bp3-label.bp3-inline .bp3-input-group .bp3-input{
      margin-left:0; }
    label.bp3-label.bp3-inline.bp3-large{
      line-height:40px; }
  label.bp3-label:not(.bp3-inline) .bp3-popover-target{
    display:block; }
  .bp3-dark label.bp3-label{
    color:#f5f8fa; }
    .bp3-dark label.bp3-label.bp3-disabled,
    .bp3-dark label.bp3-label.bp3-disabled .bp3-text-muted{
      color:rgba(167, 182, 194, 0.6); }
.bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button{
  -webkit-box-flex:1;
      -ms-flex:1 1 14px;
          flex:1 1 14px;
  min-height:0;
  padding:0;
  width:30px; }
  .bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button:first-child{
    border-radius:0 3px 0 0; }
  .bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button:last-child{
    border-radius:0 0 3px 0; }

.bp3-numeric-input .bp3-button-group.bp3-vertical:first-child > .bp3-button:first-child{
  border-radius:3px 0 0 0; }

.bp3-numeric-input .bp3-button-group.bp3-vertical:first-child > .bp3-button:last-child{
  border-radius:0 0 0 3px; }

.bp3-numeric-input.bp3-large .bp3-button-group.bp3-vertical > .bp3-button{
  width:40px; }

form{
  display:block; }
.bp3-html-select select,
.bp3-select select{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  border:none;
  border-radius:3px;
  cursor:pointer;
  font-size:14px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  padding:5px 10px;
  text-align:left;
  vertical-align:middle;
  background-color:#f5f8fa;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
  color:#182026;
  -moz-appearance:none;
  -webkit-appearance:none;
  border-radius:3px;
  height:30px;
  padding:0 25px 0 10px;
  width:100%; }
  .bp3-html-select select > *, .bp3-select select > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-html-select select > .bp3-fill, .bp3-select select > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-html-select select::before,
  .bp3-select select::before, .bp3-html-select select > *, .bp3-select select > *{
    margin-right:7px; }
  .bp3-html-select select:empty::before,
  .bp3-select select:empty::before,
  .bp3-html-select select > :last-child,
  .bp3-select select > :last-child{
    margin-right:0; }
  .bp3-html-select select:hover,
  .bp3-select select:hover{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
  .bp3-html-select select:active,
  .bp3-select select:active, .bp3-html-select select.bp3-active,
  .bp3-select select.bp3-active{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-html-select select:disabled,
  .bp3-select select:disabled, .bp3-html-select select.bp3-disabled,
  .bp3-select select.bp3-disabled{
    background-color:rgba(206, 217, 224, 0.5);
    background-image:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    outline:none; }
    .bp3-html-select select:disabled.bp3-active,
    .bp3-select select:disabled.bp3-active, .bp3-html-select select:disabled.bp3-active:hover,
    .bp3-select select:disabled.bp3-active:hover, .bp3-html-select select.bp3-disabled.bp3-active,
    .bp3-select select.bp3-disabled.bp3-active, .bp3-html-select select.bp3-disabled.bp3-active:hover,
    .bp3-select select.bp3-disabled.bp3-active:hover{
      background:rgba(206, 217, 224, 0.7); }

.bp3-html-select.bp3-minimal select,
.bp3-select.bp3-minimal select{
  background:none;
  -webkit-box-shadow:none;
          box-shadow:none; }
  .bp3-html-select.bp3-minimal select:hover,
  .bp3-select.bp3-minimal select:hover{
    background:rgba(167, 182, 194, 0.3);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:#182026;
    text-decoration:none; }
  .bp3-html-select.bp3-minimal select:active,
  .bp3-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal select.bp3-active,
  .bp3-select.bp3-minimal select.bp3-active{
    background:rgba(115, 134, 148, 0.3);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:#182026; }
  .bp3-html-select.bp3-minimal select:disabled,
  .bp3-select.bp3-minimal select:disabled, .bp3-html-select.bp3-minimal select:disabled:hover,
  .bp3-select.bp3-minimal select:disabled:hover, .bp3-html-select.bp3-minimal select.bp3-disabled,
  .bp3-select.bp3-minimal select.bp3-disabled, .bp3-html-select.bp3-minimal select.bp3-disabled:hover,
  .bp3-select.bp3-minimal select.bp3-disabled:hover{
    background:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
    .bp3-html-select.bp3-minimal select:disabled.bp3-active,
    .bp3-select.bp3-minimal select:disabled.bp3-active, .bp3-html-select.bp3-minimal select:disabled:hover.bp3-active,
    .bp3-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-html-select.bp3-minimal select.bp3-disabled.bp3-active,
    .bp3-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-disabled:hover.bp3-active,
    .bp3-select.bp3-minimal select.bp3-disabled:hover.bp3-active{
      background:rgba(115, 134, 148, 0.3); }
  .bp3-dark .bp3-html-select.bp3-minimal select, .bp3-html-select.bp3-minimal .bp3-dark select,
  .bp3-dark .bp3-select.bp3-minimal select, .bp3-select.bp3-minimal .bp3-dark select{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:inherit; }
    .bp3-dark .bp3-html-select.bp3-minimal select:hover, .bp3-html-select.bp3-minimal .bp3-dark select:hover,
    .bp3-dark .bp3-select.bp3-minimal select:hover, .bp3-select.bp3-minimal .bp3-dark select:hover, .bp3-dark .bp3-html-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal .bp3-dark select:active,
    .bp3-dark .bp3-select.bp3-minimal select:active, .bp3-select.bp3-minimal .bp3-dark select:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-active,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-html-select.bp3-minimal select:hover, .bp3-html-select.bp3-minimal .bp3-dark select:hover,
    .bp3-dark .bp3-select.bp3-minimal select:hover, .bp3-select.bp3-minimal .bp3-dark select:hover{
      background:rgba(138, 155, 168, 0.15); }
    .bp3-dark .bp3-html-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal .bp3-dark select:active,
    .bp3-dark .bp3-select.bp3-minimal select:active, .bp3-select.bp3-minimal .bp3-dark select:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-active,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-active{
      background:rgba(138, 155, 168, 0.3);
      color:#f5f8fa; }
    .bp3-dark .bp3-html-select.bp3-minimal select:disabled, .bp3-html-select.bp3-minimal .bp3-dark select:disabled,
    .bp3-dark .bp3-select.bp3-minimal select:disabled, .bp3-select.bp3-minimal .bp3-dark select:disabled, .bp3-dark .bp3-html-select.bp3-minimal select:disabled:hover, .bp3-html-select.bp3-minimal .bp3-dark select:disabled:hover,
    .bp3-dark .bp3-select.bp3-minimal select:disabled:hover, .bp3-select.bp3-minimal .bp3-dark select:disabled:hover, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled:hover,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled:hover{
      background:none;
      color:rgba(167, 182, 194, 0.6);
      cursor:not-allowed; }
      .bp3-dark .bp3-html-select.bp3-minimal select:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select:disabled.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select:disabled:hover.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-select.bp3-minimal .bp3-dark select:disabled:hover.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled:hover.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled:hover.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled:hover.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled:hover.bp3-active{
        background:rgba(138, 155, 168, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-primary,
  .bp3-select.bp3-minimal select.bp3-intent-primary{
    color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover,
    .bp3-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-html-select.bp3-minimal select.bp3-intent-primary:active,
    .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover,
    .bp3-select.bp3-minimal select.bp3-intent-primary:hover{
      background:rgba(19, 124, 189, 0.15);
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:active,
    .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active{
      background:rgba(19, 124, 189, 0.3);
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled{
      background:none;
      color:rgba(16, 107, 163, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active{
        background:rgba(19, 124, 189, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
      stroke:#106ba3; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary{
      color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.2);
        color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(72, 175, 240, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-success,
  .bp3-select.bp3-minimal select.bp3-intent-success{
    color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:hover,
    .bp3-select.bp3-minimal select.bp3-intent-success:hover, .bp3-html-select.bp3-minimal select.bp3-intent-success:active,
    .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:hover,
    .bp3-select.bp3-minimal select.bp3-intent-success:hover{
      background:rgba(15, 153, 96, 0.15);
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:active,
    .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active{
      background:rgba(15, 153, 96, 0.3);
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled{
      background:none;
      color:rgba(13, 128, 80, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active{
        background:rgba(15, 153, 96, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-success .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
      stroke:#0d8050; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success{
      color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.2);
        color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(61, 204, 145, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-warning,
  .bp3-select.bp3-minimal select.bp3-intent-warning{
    color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover,
    .bp3-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-html-select.bp3-minimal select.bp3-intent-warning:active,
    .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover,
    .bp3-select.bp3-minimal select.bp3-intent-warning:hover{
      background:rgba(217, 130, 43, 0.15);
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:active,
    .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active{
      background:rgba(217, 130, 43, 0.3);
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled{
      background:none;
      color:rgba(191, 115, 38, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active{
        background:rgba(217, 130, 43, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
      stroke:#bf7326; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning{
      color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.2);
        color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(255, 179, 102, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-danger,
  .bp3-select.bp3-minimal select.bp3-intent-danger{
    color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover,
    .bp3-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-html-select.bp3-minimal select.bp3-intent-danger:active,
    .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover,
    .bp3-select.bp3-minimal select.bp3-intent-danger:hover{
      background:rgba(219, 55, 55, 0.15);
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:active,
    .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active{
      background:rgba(219, 55, 55, 0.3);
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled{
      background:none;
      color:rgba(194, 48, 48, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active{
        background:rgba(219, 55, 55, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
      stroke:#c23030; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger{
      color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.2);
        color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(255, 115, 115, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }

.bp3-html-select.bp3-large select,
.bp3-select.bp3-large select{
  font-size:16px;
  height:40px;
  padding-right:35px; }

.bp3-dark .bp3-html-select select, .bp3-dark .bp3-select select{
  background-color:#394b59;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
  color:#f5f8fa; }
  .bp3-dark .bp3-html-select select:hover, .bp3-dark .bp3-select select:hover, .bp3-dark .bp3-html-select select:active, .bp3-dark .bp3-select select:active, .bp3-dark .bp3-html-select select.bp3-active, .bp3-dark .bp3-select select.bp3-active{
    color:#f5f8fa; }
  .bp3-dark .bp3-html-select select:hover, .bp3-dark .bp3-select select:hover{
    background-color:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-html-select select:active, .bp3-dark .bp3-select select:active, .bp3-dark .bp3-html-select select.bp3-active, .bp3-dark .bp3-select select.bp3-active{
    background-color:#202b33;
    background-image:none;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-html-select select:disabled, .bp3-dark .bp3-select select:disabled, .bp3-dark .bp3-html-select select.bp3-disabled, .bp3-dark .bp3-select select.bp3-disabled{
    background-color:rgba(57, 75, 89, 0.5);
    background-image:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-html-select select:disabled.bp3-active, .bp3-dark .bp3-select select:disabled.bp3-active, .bp3-dark .bp3-html-select select.bp3-disabled.bp3-active, .bp3-dark .bp3-select select.bp3-disabled.bp3-active{
      background:rgba(57, 75, 89, 0.7); }
  .bp3-dark .bp3-html-select select .bp3-button-spinner .bp3-spinner-head, .bp3-dark .bp3-select select .bp3-button-spinner .bp3-spinner-head{
    background:rgba(16, 22, 26, 0.5);
    stroke:#8a9ba8; }

.bp3-html-select select:disabled,
.bp3-select select:disabled{
  background-color:rgba(206, 217, 224, 0.5);
  -webkit-box-shadow:none;
          box-shadow:none;
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-html-select .bp3-icon,
.bp3-select .bp3-icon, .bp3-select::after{
  color:#5c7080;
  pointer-events:none;
  position:absolute;
  right:7px;
  top:7px; }
  .bp3-html-select .bp3-disabled.bp3-icon,
  .bp3-select .bp3-disabled.bp3-icon, .bp3-disabled.bp3-select::after{
    color:rgba(92, 112, 128, 0.6); }
.bp3-html-select,
.bp3-select{
  display:inline-block;
  letter-spacing:normal;
  position:relative;
  vertical-align:middle; }
  .bp3-html-select select::-ms-expand,
  .bp3-select select::-ms-expand{
    display:none; }
  .bp3-html-select .bp3-icon,
  .bp3-select .bp3-icon{
    color:#5c7080; }
    .bp3-html-select .bp3-icon:hover,
    .bp3-select .bp3-icon:hover{
      color:#182026; }
    .bp3-dark .bp3-html-select .bp3-icon, .bp3-dark
    .bp3-select .bp3-icon{
      color:#a7b6c2; }
      .bp3-dark .bp3-html-select .bp3-icon:hover, .bp3-dark
      .bp3-select .bp3-icon:hover{
        color:#f5f8fa; }
  .bp3-html-select.bp3-large::after,
  .bp3-html-select.bp3-large .bp3-icon,
  .bp3-select.bp3-large::after,
  .bp3-select.bp3-large .bp3-icon{
    right:12px;
    top:12px; }
  .bp3-html-select.bp3-fill,
  .bp3-html-select.bp3-fill select,
  .bp3-select.bp3-fill,
  .bp3-select.bp3-fill select{
    width:100%; }
  .bp3-dark .bp3-html-select option, .bp3-dark
  .bp3-select option{
    background-color:#30404d;
    color:#f5f8fa; }
  .bp3-dark .bp3-html-select option:disabled, .bp3-dark
  .bp3-select option:disabled{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-html-select::after, .bp3-dark
  .bp3-select::after{
    color:#a7b6c2; }

.bp3-select::after{
  font-family:"Icons16", sans-serif;
  font-size:16px;
  font-style:normal;
  font-weight:400;
  line-height:1;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  content:""; }
.bp3-running-text table, table.bp3-html-table{
  border-spacing:0;
  font-size:14px; }
  .bp3-running-text table th, table.bp3-html-table th,
  .bp3-running-text table td,
  table.bp3-html-table td{
    padding:11px;
    text-align:left;
    vertical-align:top; }
  .bp3-running-text table th, table.bp3-html-table th{
    color:#182026;
    font-weight:600; }
  
  .bp3-running-text table td,
  table.bp3-html-table td{
    color:#182026; }
  .bp3-running-text table tbody tr:first-child th, table.bp3-html-table tbody tr:first-child th,
  .bp3-running-text table tbody tr:first-child td,
  table.bp3-html-table tbody tr:first-child td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
  .bp3-dark .bp3-running-text table th, .bp3-running-text .bp3-dark table th, .bp3-dark table.bp3-html-table th{
    color:#f5f8fa; }
  .bp3-dark .bp3-running-text table td, .bp3-running-text .bp3-dark table td, .bp3-dark table.bp3-html-table td{
    color:#f5f8fa; }
  .bp3-dark .bp3-running-text table tbody tr:first-child th, .bp3-running-text .bp3-dark table tbody tr:first-child th, .bp3-dark table.bp3-html-table tbody tr:first-child th,
  .bp3-dark .bp3-running-text table tbody tr:first-child td,
  .bp3-running-text .bp3-dark table tbody tr:first-child td,
  .bp3-dark table.bp3-html-table tbody tr:first-child td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15); }

table.bp3-html-table.bp3-html-table-condensed th,
table.bp3-html-table.bp3-html-table-condensed td, table.bp3-html-table.bp3-small th,
table.bp3-html-table.bp3-small td{
  padding-bottom:6px;
  padding-top:6px; }

table.bp3-html-table.bp3-html-table-striped tbody tr:nth-child(odd) td{
  background:rgba(191, 204, 214, 0.15); }

table.bp3-html-table.bp3-html-table-bordered th:not(:first-child){
  -webkit-box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-html-table-bordered tbody tr td{
  -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
  table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child){
    -webkit-box-shadow:inset 1px 1px 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 1px 1px 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td{
  -webkit-box-shadow:none;
          box-shadow:none; }
  table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td:not(:first-child){
    -webkit-box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-interactive tbody tr:hover td{
  background-color:rgba(191, 204, 214, 0.3);
  cursor:pointer; }

table.bp3-html-table.bp3-interactive tbody tr:active td{
  background-color:rgba(191, 204, 214, 0.4); }

.bp3-dark table.bp3-html-table{ }
  .bp3-dark table.bp3-html-table.bp3-html-table-striped tbody tr:nth-child(odd) td{
    background:rgba(92, 112, 128, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered th:not(:first-child){
    -webkit-box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15); }
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child){
      -webkit-box-shadow:inset 1px 1px 0 0 rgba(255, 255, 255, 0.15);
              box-shadow:inset 1px 1px 0 0 rgba(255, 255, 255, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td{
    -webkit-box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15); }
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td:first-child{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-dark table.bp3-html-table.bp3-interactive tbody tr:hover td{
    background-color:rgba(92, 112, 128, 0.3);
    cursor:pointer; }
  .bp3-dark table.bp3-html-table.bp3-interactive tbody tr:active td{
    background-color:rgba(92, 112, 128, 0.4); }

.bp3-key-combo{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center; }
  .bp3-key-combo > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-key-combo > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-key-combo::before,
  .bp3-key-combo > *{
    margin-right:5px; }
  .bp3-key-combo:empty::before,
  .bp3-key-combo > :last-child{
    margin-right:0; }

.bp3-hotkey-dialog{
  padding-bottom:0;
  top:40px; }
  .bp3-hotkey-dialog .bp3-dialog-body{
    margin:0;
    padding:0; }
  .bp3-hotkey-dialog .bp3-hotkey-label{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1; }

.bp3-hotkey-column{
  margin:auto;
  max-height:80vh;
  overflow-y:auto;
  padding:30px; }
  .bp3-hotkey-column .bp3-heading{
    margin-bottom:20px; }
    .bp3-hotkey-column .bp3-heading:not(:first-child){
      margin-top:40px; }

.bp3-hotkey{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:justify;
      -ms-flex-pack:justify;
          justify-content:space-between;
  margin-left:0;
  margin-right:0; }
  .bp3-hotkey:not(:last-child){
    margin-bottom:10px; }
.bp3-icon{
  display:inline-block;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  vertical-align:text-bottom; }
  .bp3-icon:not(:empty)::before{
    content:"" !important;
    content:unset !important; }
  .bp3-icon > svg{
    display:block; }
    .bp3-icon > svg:not([fill]){
      fill:currentColor; }

.bp3-icon.bp3-intent-primary, .bp3-icon-standard.bp3-intent-primary, .bp3-icon-large.bp3-intent-primary{
  color:#106ba3; }
  .bp3-dark .bp3-icon.bp3-intent-primary, .bp3-dark .bp3-icon-standard.bp3-intent-primary, .bp3-dark .bp3-icon-large.bp3-intent-primary{
    color:#48aff0; }

.bp3-icon.bp3-intent-success, .bp3-icon-standard.bp3-intent-success, .bp3-icon-large.bp3-intent-success{
  color:#0d8050; }
  .bp3-dark .bp3-icon.bp3-intent-success, .bp3-dark .bp3-icon-standard.bp3-intent-success, .bp3-dark .bp3-icon-large.bp3-intent-success{
    color:#3dcc91; }

.bp3-icon.bp3-intent-warning, .bp3-icon-standard.bp3-intent-warning, .bp3-icon-large.bp3-intent-warning{
  color:#bf7326; }
  .bp3-dark .bp3-icon.bp3-intent-warning, .bp3-dark .bp3-icon-standard.bp3-intent-warning, .bp3-dark .bp3-icon-large.bp3-intent-warning{
    color:#ffb366; }

.bp3-icon.bp3-intent-danger, .bp3-icon-standard.bp3-intent-danger, .bp3-icon-large.bp3-intent-danger{
  color:#c23030; }
  .bp3-dark .bp3-icon.bp3-intent-danger, .bp3-dark .bp3-icon-standard.bp3-intent-danger, .bp3-dark .bp3-icon-large.bp3-intent-danger{
    color:#ff7373; }

span.bp3-icon-standard{
  font-family:"Icons16", sans-serif;
  font-size:16px;
  font-style:normal;
  font-weight:400;
  line-height:1;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  display:inline-block; }

span.bp3-icon-large{
  font-family:"Icons20", sans-serif;
  font-size:20px;
  font-style:normal;
  font-weight:400;
  line-height:1;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  display:inline-block; }

span.bp3-icon:empty{
  font-family:"Icons20";
  font-size:inherit;
  font-style:normal;
  font-weight:400;
  line-height:1; }
  span.bp3-icon:empty::before{
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased; }

.bp3-icon-add::before{
  content:""; }

.bp3-icon-add-column-left::before{
  content:""; }

.bp3-icon-add-column-right::before{
  content:""; }

.bp3-icon-add-row-bottom::before{
  content:""; }

.bp3-icon-add-row-top::before{
  content:""; }

.bp3-icon-add-to-artifact::before{
  content:""; }

.bp3-icon-add-to-folder::before{
  content:""; }

.bp3-icon-airplane::before{
  content:""; }

.bp3-icon-align-center::before{
  content:""; }

.bp3-icon-align-justify::before{
  content:""; }

.bp3-icon-align-left::before{
  content:""; }

.bp3-icon-align-right::before{
  content:""; }

.bp3-icon-alignment-bottom::before{
  content:""; }

.bp3-icon-alignment-horizontal-center::before{
  content:""; }

.bp3-icon-alignment-left::before{
  content:""; }

.bp3-icon-alignment-right::before{
  content:""; }

.bp3-icon-alignment-top::before{
  content:""; }

.bp3-icon-alignment-vertical-center::before{
  content:""; }

.bp3-icon-annotation::before{
  content:""; }

.bp3-icon-application::before{
  content:""; }

.bp3-icon-applications::before{
  content:""; }

.bp3-icon-archive::before{
  content:""; }

.bp3-icon-arrow-bottom-left::before{
  content:"↙"; }

.bp3-icon-arrow-bottom-right::before{
  content:"↘"; }

.bp3-icon-arrow-down::before{
  content:"↓"; }

.bp3-icon-arrow-left::before{
  content:"←"; }

.bp3-icon-arrow-right::before{
  content:"→"; }

.bp3-icon-arrow-top-left::before{
  content:"↖"; }

.bp3-icon-arrow-top-right::before{
  content:"↗"; }

.bp3-icon-arrow-up::before{
  content:"↑"; }

.bp3-icon-arrows-horizontal::before{
  content:"↔"; }

.bp3-icon-arrows-vertical::before{
  content:"↕"; }

.bp3-icon-asterisk::before{
  content:"*"; }

.bp3-icon-automatic-updates::before{
  content:""; }

.bp3-icon-badge::before{
  content:""; }

.bp3-icon-ban-circle::before{
  content:""; }

.bp3-icon-bank-account::before{
  content:""; }

.bp3-icon-barcode::before{
  content:""; }

.bp3-icon-blank::before{
  content:""; }

.bp3-icon-blocked-person::before{
  content:""; }

.bp3-icon-bold::before{
  content:""; }

.bp3-icon-book::before{
  content:""; }

.bp3-icon-bookmark::before{
  content:""; }

.bp3-icon-box::before{
  content:""; }

.bp3-icon-briefcase::before{
  content:""; }

.bp3-icon-bring-data::before{
  content:""; }

.bp3-icon-build::before{
  content:""; }

.bp3-icon-calculator::before{
  content:""; }

.bp3-icon-calendar::before{
  content:""; }

.bp3-icon-camera::before{
  content:""; }

.bp3-icon-caret-down::before{
  content:"⌄"; }

.bp3-icon-caret-left::before{
  content:"〈"; }

.bp3-icon-caret-right::before{
  content:"〉"; }

.bp3-icon-caret-up::before{
  content:"⌃"; }

.bp3-icon-cell-tower::before{
  content:""; }

.bp3-icon-changes::before{
  content:""; }

.bp3-icon-chart::before{
  content:""; }

.bp3-icon-chat::before{
  content:""; }

.bp3-icon-chevron-backward::before{
  content:""; }

.bp3-icon-chevron-down::before{
  content:""; }

.bp3-icon-chevron-forward::before{
  content:""; }

.bp3-icon-chevron-left::before{
  content:""; }

.bp3-icon-chevron-right::before{
  content:""; }

.bp3-icon-chevron-up::before{
  content:""; }

.bp3-icon-circle::before{
  content:""; }

.bp3-icon-circle-arrow-down::before{
  content:""; }

.bp3-icon-circle-arrow-left::before{
  content:""; }

.bp3-icon-circle-arrow-right::before{
  content:""; }

.bp3-icon-circle-arrow-up::before{
  content:""; }

.bp3-icon-citation::before{
  content:""; }

.bp3-icon-clean::before{
  content:""; }

.bp3-icon-clipboard::before{
  content:""; }

.bp3-icon-cloud::before{
  content:"☁"; }

.bp3-icon-cloud-download::before{
  content:""; }

.bp3-icon-cloud-upload::before{
  content:""; }

.bp3-icon-code::before{
  content:""; }

.bp3-icon-code-block::before{
  content:""; }

.bp3-icon-cog::before{
  content:""; }

.bp3-icon-collapse-all::before{
  content:""; }

.bp3-icon-column-layout::before{
  content:""; }

.bp3-icon-comment::before{
  content:""; }

.bp3-icon-comparison::before{
  content:""; }

.bp3-icon-compass::before{
  content:""; }

.bp3-icon-compressed::before{
  content:""; }

.bp3-icon-confirm::before{
  content:""; }

.bp3-icon-console::before{
  content:""; }

.bp3-icon-contrast::before{
  content:""; }

.bp3-icon-control::before{
  content:""; }

.bp3-icon-credit-card::before{
  content:""; }

.bp3-icon-cross::before{
  content:"✗"; }

.bp3-icon-crown::before{
  content:""; }

.bp3-icon-cube::before{
  content:""; }

.bp3-icon-cube-add::before{
  content:""; }

.bp3-icon-cube-remove::before{
  content:""; }

.bp3-icon-curved-range-chart::before{
  content:""; }

.bp3-icon-cut::before{
  content:""; }

.bp3-icon-dashboard::before{
  content:""; }

.bp3-icon-data-lineage::before{
  content:""; }

.bp3-icon-database::before{
  content:""; }

.bp3-icon-delete::before{
  content:""; }

.bp3-icon-delta::before{
  content:"Δ"; }

.bp3-icon-derive-column::before{
  content:""; }

.bp3-icon-desktop::before{
  content:""; }

.bp3-icon-diagnosis::before{
  content:""; }

.bp3-icon-diagram-tree::before{
  content:""; }

.bp3-icon-direction-left::before{
  content:""; }

.bp3-icon-direction-right::before{
  content:""; }

.bp3-icon-disable::before{
  content:""; }

.bp3-icon-document::before{
  content:""; }

.bp3-icon-document-open::before{
  content:""; }

.bp3-icon-document-share::before{
  content:""; }

.bp3-icon-dollar::before{
  content:"$"; }

.bp3-icon-dot::before{
  content:"•"; }

.bp3-icon-double-caret-horizontal::before{
  content:""; }

.bp3-icon-double-caret-vertical::before{
  content:""; }

.bp3-icon-double-chevron-down::before{
  content:""; }

.bp3-icon-double-chevron-left::before{
  content:""; }

.bp3-icon-double-chevron-right::before{
  content:""; }

.bp3-icon-double-chevron-up::before{
  content:""; }

.bp3-icon-doughnut-chart::before{
  content:""; }

.bp3-icon-download::before{
  content:""; }

.bp3-icon-drag-handle-horizontal::before{
  content:""; }

.bp3-icon-drag-handle-vertical::before{
  content:""; }

.bp3-icon-draw::before{
  content:""; }

.bp3-icon-drive-time::before{
  content:""; }

.bp3-icon-duplicate::before{
  content:""; }

.bp3-icon-edit::before{
  content:"✎"; }

.bp3-icon-eject::before{
  content:"⏏"; }

.bp3-icon-endorsed::before{
  content:""; }

.bp3-icon-envelope::before{
  content:"✉"; }

.bp3-icon-equals::before{
  content:""; }

.bp3-icon-eraser::before{
  content:""; }

.bp3-icon-error::before{
  content:""; }

.bp3-icon-euro::before{
  content:"€"; }

.bp3-icon-exchange::before{
  content:""; }

.bp3-icon-exclude-row::before{
  content:""; }

.bp3-icon-expand-all::before{
  content:""; }

.bp3-icon-export::before{
  content:""; }

.bp3-icon-eye-off::before{
  content:""; }

.bp3-icon-eye-on::before{
  content:""; }

.bp3-icon-eye-open::before{
  content:""; }

.bp3-icon-fast-backward::before{
  content:""; }

.bp3-icon-fast-forward::before{
  content:""; }

.bp3-icon-feed::before{
  content:""; }

.bp3-icon-feed-subscribed::before{
  content:""; }

.bp3-icon-film::before{
  content:""; }

.bp3-icon-filter::before{
  content:""; }

.bp3-icon-filter-keep::before{
  content:""; }

.bp3-icon-filter-list::before{
  content:""; }

.bp3-icon-filter-open::before{
  content:""; }

.bp3-icon-filter-remove::before{
  content:""; }

.bp3-icon-flag::before{
  content:"⚑"; }

.bp3-icon-flame::before{
  content:""; }

.bp3-icon-flash::before{
  content:""; }

.bp3-icon-floppy-disk::before{
  content:""; }

.bp3-icon-flow-branch::before{
  content:""; }

.bp3-icon-flow-end::before{
  content:""; }

.bp3-icon-flow-linear::before{
  content:""; }

.bp3-icon-flow-review::before{
  content:""; }

.bp3-icon-flow-review-branch::before{
  content:""; }

.bp3-icon-flows::before{
  content:""; }

.bp3-icon-folder-close::before{
  content:""; }

.bp3-icon-folder-new::before{
  content:""; }

.bp3-icon-folder-open::before{
  content:""; }

.bp3-icon-folder-shared::before{
  content:""; }

.bp3-icon-folder-shared-open::before{
  content:""; }

.bp3-icon-follower::before{
  content:""; }

.bp3-icon-following::before{
  content:""; }

.bp3-icon-font::before{
  content:""; }

.bp3-icon-fork::before{
  content:""; }

.bp3-icon-form::before{
  content:""; }

.bp3-icon-full-circle::before{
  content:""; }

.bp3-icon-full-stacked-chart::before{
  content:""; }

.bp3-icon-fullscreen::before{
  content:""; }

.bp3-icon-function::before{
  content:""; }

.bp3-icon-gantt-chart::before{
  content:""; }

.bp3-icon-geolocation::before{
  content:""; }

.bp3-icon-geosearch::before{
  content:""; }

.bp3-icon-git-branch::before{
  content:""; }

.bp3-icon-git-commit::before{
  content:""; }

.bp3-icon-git-merge::before{
  content:""; }

.bp3-icon-git-new-branch::before{
  content:""; }

.bp3-icon-git-pull::before{
  content:""; }

.bp3-icon-git-push::before{
  content:""; }

.bp3-icon-git-repo::before{
  content:""; }

.bp3-icon-glass::before{
  content:""; }

.bp3-icon-globe::before{
  content:""; }

.bp3-icon-globe-network::before{
  content:""; }

.bp3-icon-graph::before{
  content:""; }

.bp3-icon-graph-remove::before{
  content:""; }

.bp3-icon-greater-than::before{
  content:""; }

.bp3-icon-greater-than-or-equal-to::before{
  content:""; }

.bp3-icon-grid::before{
  content:""; }

.bp3-icon-grid-view::before{
  content:""; }

.bp3-icon-group-objects::before{
  content:""; }

.bp3-icon-grouped-bar-chart::before{
  content:""; }

.bp3-icon-hand::before{
  content:""; }

.bp3-icon-hand-down::before{
  content:""; }

.bp3-icon-hand-left::before{
  content:""; }

.bp3-icon-hand-right::before{
  content:""; }

.bp3-icon-hand-up::before{
  content:""; }

.bp3-icon-header::before{
  content:""; }

.bp3-icon-header-one::before{
  content:""; }

.bp3-icon-header-two::before{
  content:""; }

.bp3-icon-headset::before{
  content:""; }

.bp3-icon-heart::before{
  content:"♥"; }

.bp3-icon-heart-broken::before{
  content:""; }

.bp3-icon-heat-grid::before{
  content:""; }

.bp3-icon-heatmap::before{
  content:""; }

.bp3-icon-help::before{
  content:"?"; }

.bp3-icon-helper-management::before{
  content:""; }

.bp3-icon-highlight::before{
  content:""; }

.bp3-icon-history::before{
  content:""; }

.bp3-icon-home::before{
  content:"⌂"; }

.bp3-icon-horizontal-bar-chart::before{
  content:""; }

.bp3-icon-horizontal-bar-chart-asc::before{
  content:""; }

.bp3-icon-horizontal-bar-chart-desc::before{
  content:""; }

.bp3-icon-horizontal-distribution::before{
  content:""; }

.bp3-icon-id-number::before{
  content:""; }

.bp3-icon-image-rotate-left::before{
  content:""; }

.bp3-icon-image-rotate-right::before{
  content:""; }

.bp3-icon-import::before{
  content:""; }

.bp3-icon-inbox::before{
  content:""; }

.bp3-icon-inbox-filtered::before{
  content:""; }

.bp3-icon-inbox-geo::before{
  content:""; }

.bp3-icon-inbox-search::before{
  content:""; }

.bp3-icon-inbox-update::before{
  content:""; }

.bp3-icon-info-sign::before{
  content:"ℹ"; }

.bp3-icon-inheritance::before{
  content:""; }

.bp3-icon-inner-join::before{
  content:""; }

.bp3-icon-insert::before{
  content:""; }

.bp3-icon-intersection::before{
  content:""; }

.bp3-icon-ip-address::before{
  content:""; }

.bp3-icon-issue::before{
  content:""; }

.bp3-icon-issue-closed::before{
  content:""; }

.bp3-icon-issue-new::before{
  content:""; }

.bp3-icon-italic::before{
  content:""; }

.bp3-icon-join-table::before{
  content:""; }

.bp3-icon-key::before{
  content:""; }

.bp3-icon-key-backspace::before{
  content:""; }

.bp3-icon-key-command::before{
  content:""; }

.bp3-icon-key-control::before{
  content:""; }

.bp3-icon-key-delete::before{
  content:""; }

.bp3-icon-key-enter::before{
  content:""; }

.bp3-icon-key-escape::before{
  content:""; }

.bp3-icon-key-option::before{
  content:""; }

.bp3-icon-key-shift::before{
  content:""; }

.bp3-icon-key-tab::before{
  content:""; }

.bp3-icon-known-vehicle::before{
  content:""; }

.bp3-icon-lab-test::before{
  content:""; }

.bp3-icon-label::before{
  content:""; }

.bp3-icon-layer::before{
  content:""; }

.bp3-icon-layers::before{
  content:""; }

.bp3-icon-layout::before{
  content:""; }

.bp3-icon-layout-auto::before{
  content:""; }

.bp3-icon-layout-balloon::before{
  content:""; }

.bp3-icon-layout-circle::before{
  content:""; }

.bp3-icon-layout-grid::before{
  content:""; }

.bp3-icon-layout-group-by::before{
  content:""; }

.bp3-icon-layout-hierarchy::before{
  content:""; }

.bp3-icon-layout-linear::before{
  content:""; }

.bp3-icon-layout-skew-grid::before{
  content:""; }

.bp3-icon-layout-sorted-clusters::before{
  content:""; }

.bp3-icon-learning::before{
  content:""; }

.bp3-icon-left-join::before{
  content:""; }

.bp3-icon-less-than::before{
  content:""; }

.bp3-icon-less-than-or-equal-to::before{
  content:""; }

.bp3-icon-lifesaver::before{
  content:""; }

.bp3-icon-lightbulb::before{
  content:""; }

.bp3-icon-link::before{
  content:""; }

.bp3-icon-list::before{
  content:"☰"; }

.bp3-icon-list-columns::before{
  content:""; }

.bp3-icon-list-detail-view::before{
  content:""; }

.bp3-icon-locate::before{
  content:""; }

.bp3-icon-lock::before{
  content:""; }

.bp3-icon-log-in::before{
  content:""; }

.bp3-icon-log-out::before{
  content:""; }

.bp3-icon-manual::before{
  content:""; }

.bp3-icon-manually-entered-data::before{
  content:""; }

.bp3-icon-map::before{
  content:""; }

.bp3-icon-map-create::before{
  content:""; }

.bp3-icon-map-marker::before{
  content:""; }

.bp3-icon-maximize::before{
  content:""; }

.bp3-icon-media::before{
  content:""; }

.bp3-icon-menu::before{
  content:""; }

.bp3-icon-menu-closed::before{
  content:""; }

.bp3-icon-menu-open::before{
  content:""; }

.bp3-icon-merge-columns::before{
  content:""; }

.bp3-icon-merge-links::before{
  content:""; }

.bp3-icon-minimize::before{
  content:""; }

.bp3-icon-minus::before{
  content:"−"; }

.bp3-icon-mobile-phone::before{
  content:""; }

.bp3-icon-mobile-video::before{
  content:""; }

.bp3-icon-moon::before{
  content:""; }

.bp3-icon-more::before{
  content:""; }

.bp3-icon-mountain::before{
  content:""; }

.bp3-icon-move::before{
  content:""; }

.bp3-icon-mugshot::before{
  content:""; }

.bp3-icon-multi-select::before{
  content:""; }

.bp3-icon-music::before{
  content:""; }

.bp3-icon-new-drawing::before{
  content:""; }

.bp3-icon-new-grid-item::before{
  content:""; }

.bp3-icon-new-layer::before{
  content:""; }

.bp3-icon-new-layers::before{
  content:""; }

.bp3-icon-new-link::before{
  content:""; }

.bp3-icon-new-object::before{
  content:""; }

.bp3-icon-new-person::before{
  content:""; }

.bp3-icon-new-prescription::before{
  content:""; }

.bp3-icon-new-text-box::before{
  content:""; }

.bp3-icon-ninja::before{
  content:""; }

.bp3-icon-not-equal-to::before{
  content:""; }

.bp3-icon-notifications::before{
  content:""; }

.bp3-icon-notifications-updated::before{
  content:""; }

.bp3-icon-numbered-list::before{
  content:""; }

.bp3-icon-numerical::before{
  content:""; }

.bp3-icon-office::before{
  content:""; }

.bp3-icon-offline::before{
  content:""; }

.bp3-icon-oil-field::before{
  content:""; }

.bp3-icon-one-column::before{
  content:""; }

.bp3-icon-outdated::before{
  content:""; }

.bp3-icon-page-layout::before{
  content:""; }

.bp3-icon-panel-stats::before{
  content:""; }

.bp3-icon-panel-table::before{
  content:""; }

.bp3-icon-paperclip::before{
  content:""; }

.bp3-icon-paragraph::before{
  content:""; }

.bp3-icon-path::before{
  content:""; }

.bp3-icon-path-search::before{
  content:""; }

.bp3-icon-pause::before{
  content:""; }

.bp3-icon-people::before{
  content:""; }

.bp3-icon-percentage::before{
  content:""; }

.bp3-icon-person::before{
  content:""; }

.bp3-icon-phone::before{
  content:"☎"; }

.bp3-icon-pie-chart::before{
  content:""; }

.bp3-icon-pin::before{
  content:""; }

.bp3-icon-pivot::before{
  content:""; }

.bp3-icon-pivot-table::before{
  content:""; }

.bp3-icon-play::before{
  content:""; }

.bp3-icon-plus::before{
  content:"+"; }

.bp3-icon-polygon-filter::before{
  content:""; }

.bp3-icon-power::before{
  content:""; }

.bp3-icon-predictive-analysis::before{
  content:""; }

.bp3-icon-prescription::before{
  content:""; }

.bp3-icon-presentation::before{
  content:""; }

.bp3-icon-print::before{
  content:"⎙"; }

.bp3-icon-projects::before{
  content:""; }

.bp3-icon-properties::before{
  content:""; }

.bp3-icon-property::before{
  content:""; }

.bp3-icon-publish-function::before{
  content:""; }

.bp3-icon-pulse::before{
  content:""; }

.bp3-icon-random::before{
  content:""; }

.bp3-icon-record::before{
  content:""; }

.bp3-icon-redo::before{
  content:""; }

.bp3-icon-refresh::before{
  content:""; }

.bp3-icon-regression-chart::before{
  content:""; }

.bp3-icon-remove::before{
  content:""; }

.bp3-icon-remove-column::before{
  content:""; }

.bp3-icon-remove-column-left::before{
  content:""; }

.bp3-icon-remove-column-right::before{
  content:""; }

.bp3-icon-remove-row-bottom::before{
  content:""; }

.bp3-icon-remove-row-top::before{
  content:""; }

.bp3-icon-repeat::before{
  content:""; }

.bp3-icon-reset::before{
  content:""; }

.bp3-icon-resolve::before{
  content:""; }

.bp3-icon-rig::before{
  content:""; }

.bp3-icon-right-join::before{
  content:""; }

.bp3-icon-ring::before{
  content:""; }

.bp3-icon-rotate-document::before{
  content:""; }

.bp3-icon-rotate-page::before{
  content:""; }

.bp3-icon-satellite::before{
  content:""; }

.bp3-icon-saved::before{
  content:""; }

.bp3-icon-scatter-plot::before{
  content:""; }

.bp3-icon-search::before{
  content:""; }

.bp3-icon-search-around::before{
  content:""; }

.bp3-icon-search-template::before{
  content:""; }

.bp3-icon-search-text::before{
  content:""; }

.bp3-icon-segmented-control::before{
  content:""; }

.bp3-icon-select::before{
  content:""; }

.bp3-icon-selection::before{
  content:"⦿"; }

.bp3-icon-send-to::before{
  content:""; }

.bp3-icon-send-to-graph::before{
  content:""; }

.bp3-icon-send-to-map::before{
  content:""; }

.bp3-icon-series-add::before{
  content:""; }

.bp3-icon-series-configuration::before{
  content:""; }

.bp3-icon-series-derived::before{
  content:""; }

.bp3-icon-series-filtered::before{
  content:""; }

.bp3-icon-series-search::before{
  content:""; }

.bp3-icon-settings::before{
  content:""; }

.bp3-icon-share::before{
  content:""; }

.bp3-icon-shield::before{
  content:""; }

.bp3-icon-shop::before{
  content:""; }

.bp3-icon-shopping-cart::before{
  content:""; }

.bp3-icon-signal-search::before{
  content:""; }

.bp3-icon-sim-card::before{
  content:""; }

.bp3-icon-slash::before{
  content:""; }

.bp3-icon-small-cross::before{
  content:""; }

.bp3-icon-small-minus::before{
  content:""; }

.bp3-icon-small-plus::before{
  content:""; }

.bp3-icon-small-tick::before{
  content:""; }

.bp3-icon-snowflake::before{
  content:""; }

.bp3-icon-social-media::before{
  content:""; }

.bp3-icon-sort::before{
  content:""; }

.bp3-icon-sort-alphabetical::before{
  content:""; }

.bp3-icon-sort-alphabetical-desc::before{
  content:""; }

.bp3-icon-sort-asc::before{
  content:""; }

.bp3-icon-sort-desc::before{
  content:""; }

.bp3-icon-sort-numerical::before{
  content:""; }

.bp3-icon-sort-numerical-desc::before{
  content:""; }

.bp3-icon-split-columns::before{
  content:""; }

.bp3-icon-square::before{
  content:""; }

.bp3-icon-stacked-chart::before{
  content:""; }

.bp3-icon-star::before{
  content:"★"; }

.bp3-icon-star-empty::before{
  content:"☆"; }

.bp3-icon-step-backward::before{
  content:""; }

.bp3-icon-step-chart::before{
  content:""; }

.bp3-icon-step-forward::before{
  content:""; }

.bp3-icon-stop::before{
  content:""; }

.bp3-icon-stopwatch::before{
  content:""; }

.bp3-icon-strikethrough::before{
  content:""; }

.bp3-icon-style::before{
  content:""; }

.bp3-icon-swap-horizontal::before{
  content:""; }

.bp3-icon-swap-vertical::before{
  content:""; }

.bp3-icon-symbol-circle::before{
  content:""; }

.bp3-icon-symbol-cross::before{
  content:""; }

.bp3-icon-symbol-diamond::before{
  content:""; }

.bp3-icon-symbol-square::before{
  content:""; }

.bp3-icon-symbol-triangle-down::before{
  content:""; }

.bp3-icon-symbol-triangle-up::before{
  content:""; }

.bp3-icon-tag::before{
  content:""; }

.bp3-icon-take-action::before{
  content:""; }

.bp3-icon-taxi::before{
  content:""; }

.bp3-icon-text-highlight::before{
  content:""; }

.bp3-icon-th::before{
  content:""; }

.bp3-icon-th-derived::before{
  content:""; }

.bp3-icon-th-disconnect::before{
  content:""; }

.bp3-icon-th-filtered::before{
  content:""; }

.bp3-icon-th-list::before{
  content:""; }

.bp3-icon-thumbs-down::before{
  content:""; }

.bp3-icon-thumbs-up::before{
  content:""; }

.bp3-icon-tick::before{
  content:"✓"; }

.bp3-icon-tick-circle::before{
  content:""; }

.bp3-icon-time::before{
  content:"⏲"; }

.bp3-icon-timeline-area-chart::before{
  content:""; }

.bp3-icon-timeline-bar-chart::before{
  content:""; }

.bp3-icon-timeline-events::before{
  content:""; }

.bp3-icon-timeline-line-chart::before{
  content:""; }

.bp3-icon-tint::before{
  content:""; }

.bp3-icon-torch::before{
  content:""; }

.bp3-icon-tractor::before{
  content:""; }

.bp3-icon-train::before{
  content:""; }

.bp3-icon-translate::before{
  content:""; }

.bp3-icon-trash::before{
  content:""; }

.bp3-icon-tree::before{
  content:""; }

.bp3-icon-trending-down::before{
  content:""; }

.bp3-icon-trending-up::before{
  content:""; }

.bp3-icon-truck::before{
  content:""; }

.bp3-icon-two-columns::before{
  content:""; }

.bp3-icon-unarchive::before{
  content:""; }

.bp3-icon-underline::before{
  content:"⎁"; }

.bp3-icon-undo::before{
  content:"⎌"; }

.bp3-icon-ungroup-objects::before{
  content:""; }

.bp3-icon-unknown-vehicle::before{
  content:""; }

.bp3-icon-unlock::before{
  content:""; }

.bp3-icon-unpin::before{
  content:""; }

.bp3-icon-unresolve::before{
  content:""; }

.bp3-icon-updated::before{
  content:""; }

.bp3-icon-upload::before{
  content:""; }

.bp3-icon-user::before{
  content:""; }

.bp3-icon-variable::before{
  content:""; }

.bp3-icon-vertical-bar-chart-asc::before{
  content:""; }

.bp3-icon-vertical-bar-chart-desc::before{
  content:""; }

.bp3-icon-vertical-distribution::before{
  content:""; }

.bp3-icon-video::before{
  content:""; }

.bp3-icon-volume-down::before{
  content:""; }

.bp3-icon-volume-off::before{
  content:""; }

.bp3-icon-volume-up::before{
  content:""; }

.bp3-icon-walk::before{
  content:""; }

.bp3-icon-warning-sign::before{
  content:""; }

.bp3-icon-waterfall-chart::before{
  content:""; }

.bp3-icon-widget::before{
  content:""; }

.bp3-icon-widget-button::before{
  content:""; }

.bp3-icon-widget-footer::before{
  content:""; }

.bp3-icon-widget-header::before{
  content:""; }

.bp3-icon-wrench::before{
  content:""; }

.bp3-icon-zoom-in::before{
  content:""; }

.bp3-icon-zoom-out::before{
  content:""; }

.bp3-icon-zoom-to-fit::before{
  content:""; }
.bp3-submenu > .bp3-popover-wrapper{
  display:block; }

.bp3-submenu .bp3-popover-target{
  display:block; }
  .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{ }

.bp3-submenu.bp3-popover{
  -webkit-box-shadow:none;
          box-shadow:none;
  padding:0 5px; }
  .bp3-submenu.bp3-popover > .bp3-popover-content{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-submenu.bp3-popover, .bp3-submenu.bp3-popover.bp3-dark{
    -webkit-box-shadow:none;
            box-shadow:none; }
    .bp3-dark .bp3-submenu.bp3-popover > .bp3-popover-content, .bp3-submenu.bp3-popover.bp3-dark > .bp3-popover-content{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
.bp3-menu{
  background:#ffffff;
  border-radius:3px;
  color:#182026;
  list-style:none;
  margin:0;
  min-width:180px;
  padding:5px;
  text-align:left; }

.bp3-menu-divider{
  border-top:1px solid rgba(16, 22, 26, 0.15);
  display:block;
  margin:5px; }
  .bp3-dark .bp3-menu-divider{
    border-top-color:rgba(255, 255, 255, 0.15); }

.bp3-menu-item{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  border-radius:2px;
  color:inherit;
  line-height:20px;
  padding:5px 7px;
  text-decoration:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-menu-item > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-menu-item > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-menu-item::before,
  .bp3-menu-item > *{
    margin-right:7px; }
  .bp3-menu-item:empty::before,
  .bp3-menu-item > :last-child{
    margin-right:0; }
  .bp3-menu-item > .bp3-fill{
    word-break:break-word; }
  .bp3-menu-item:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
    background-color:rgba(167, 182, 194, 0.3);
    cursor:pointer;
    text-decoration:none; }
  .bp3-menu-item.bp3-disabled{
    background-color:inherit;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  .bp3-dark .bp3-menu-item{
    color:inherit; }
    .bp3-dark .bp3-menu-item:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
      background-color:rgba(138, 155, 168, 0.15);
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-disabled{
      background-color:inherit;
      color:rgba(167, 182, 194, 0.6); }
  .bp3-menu-item.bp3-intent-primary{
    color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-primary::before, .bp3-menu-item.bp3-intent-primary::after,
    .bp3-menu-item.bp3-intent-primary .bp3-menu-item-label{
      color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-menu-item.bp3-intent-primary.bp3-active{
      background-color:#137cbd; }
    .bp3-menu-item.bp3-intent-primary:active{
      background-color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-menu-item.bp3-intent-primary:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-menu-item.bp3-intent-primary:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-primary:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-primary:active, .bp3-menu-item.bp3-intent-primary:active::before, .bp3-menu-item.bp3-intent-primary:active::after,
    .bp3-menu-item.bp3-intent-primary:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-primary.bp3-active, .bp3-menu-item.bp3-intent-primary.bp3-active::before, .bp3-menu-item.bp3-intent-primary.bp3-active::after,
    .bp3-menu-item.bp3-intent-primary.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-success{
    color:#0d8050; }
    .bp3-menu-item.bp3-intent-success .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-success::before, .bp3-menu-item.bp3-intent-success::after,
    .bp3-menu-item.bp3-intent-success .bp3-menu-item-label{
      color:#0d8050; }
    .bp3-menu-item.bp3-intent-success:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-menu-item.bp3-intent-success.bp3-active{
      background-color:#0f9960; }
    .bp3-menu-item.bp3-intent-success:active{
      background-color:#0d8050; }
    .bp3-menu-item.bp3-intent-success:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-menu-item.bp3-intent-success:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-menu-item.bp3-intent-success:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-success:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-success:active, .bp3-menu-item.bp3-intent-success:active::before, .bp3-menu-item.bp3-intent-success:active::after,
    .bp3-menu-item.bp3-intent-success:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-success.bp3-active, .bp3-menu-item.bp3-intent-success.bp3-active::before, .bp3-menu-item.bp3-intent-success.bp3-active::after,
    .bp3-menu-item.bp3-intent-success.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-warning{
    color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-warning::before, .bp3-menu-item.bp3-intent-warning::after,
    .bp3-menu-item.bp3-intent-warning .bp3-menu-item-label{
      color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-menu-item.bp3-intent-warning.bp3-active{
      background-color:#d9822b; }
    .bp3-menu-item.bp3-intent-warning:active{
      background-color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-menu-item.bp3-intent-warning:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-menu-item.bp3-intent-warning:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-warning:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-warning:active, .bp3-menu-item.bp3-intent-warning:active::before, .bp3-menu-item.bp3-intent-warning:active::after,
    .bp3-menu-item.bp3-intent-warning:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-warning.bp3-active, .bp3-menu-item.bp3-intent-warning.bp3-active::before, .bp3-menu-item.bp3-intent-warning.bp3-active::after,
    .bp3-menu-item.bp3-intent-warning.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-danger{
    color:#c23030; }
    .bp3-menu-item.bp3-intent-danger .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-danger::before, .bp3-menu-item.bp3-intent-danger::after,
    .bp3-menu-item.bp3-intent-danger .bp3-menu-item-label{
      color:#c23030; }
    .bp3-menu-item.bp3-intent-danger:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-menu-item.bp3-intent-danger.bp3-active{
      background-color:#db3737; }
    .bp3-menu-item.bp3-intent-danger:active{
      background-color:#c23030; }
    .bp3-menu-item.bp3-intent-danger:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-menu-item.bp3-intent-danger:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-menu-item.bp3-intent-danger:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-danger:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-danger:active, .bp3-menu-item.bp3-intent-danger:active::before, .bp3-menu-item.bp3-intent-danger:active::after,
    .bp3-menu-item.bp3-intent-danger:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-danger.bp3-active, .bp3-menu-item.bp3-intent-danger.bp3-active::before, .bp3-menu-item.bp3-intent-danger.bp3-active::after,
    .bp3-menu-item.bp3-intent-danger.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item::before{
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-style:normal;
    font-weight:400;
    line-height:1;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    margin-right:7px; }
  .bp3-menu-item::before,
  .bp3-menu-item > .bp3-icon{
    color:#5c7080;
    margin-top:2px; }
  .bp3-menu-item .bp3-menu-item-label{
    color:#5c7080; }
  .bp3-menu-item:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
    color:inherit; }
  .bp3-menu-item.bp3-active, .bp3-menu-item:active{
    background-color:rgba(115, 134, 148, 0.3); }
  .bp3-menu-item.bp3-disabled{
    background-color:inherit !important;
    color:rgba(92, 112, 128, 0.6) !important;
    cursor:not-allowed !important;
    outline:none !important; }
    .bp3-menu-item.bp3-disabled::before,
    .bp3-menu-item.bp3-disabled > .bp3-icon,
    .bp3-menu-item.bp3-disabled .bp3-menu-item-label{
      color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-large .bp3-menu-item{
    font-size:16px;
    line-height:22px;
    padding:9px 7px; }
    .bp3-large .bp3-menu-item .bp3-icon{
      margin-top:3px; }
    .bp3-large .bp3-menu-item::before{
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-style:normal;
      font-weight:400;
      line-height:1;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased;
      margin-right:10px;
      margin-top:1px; }

button.bp3-menu-item{
  background:none;
  border:none;
  text-align:left;
  width:100%; }
.bp3-menu-header{
  border-top:1px solid rgba(16, 22, 26, 0.15);
  display:block;
  margin:5px;
  cursor:default;
  padding-left:2px; }
  .bp3-dark .bp3-menu-header{
    border-top-color:rgba(255, 255, 255, 0.15); }
  .bp3-menu-header:first-of-type{
    border-top:none; }
  .bp3-menu-header > h6{
    color:#182026;
    font-weight:600;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    line-height:17px;
    margin:0;
    padding:10px 7px 0 1px; }
    .bp3-dark .bp3-menu-header > h6{
      color:#f5f8fa; }
  .bp3-menu-header:first-of-type > h6{
    padding-top:0; }
  .bp3-large .bp3-menu-header > h6{
    font-size:18px;
    padding-bottom:5px;
    padding-top:15px; }
  .bp3-large .bp3-menu-header:first-of-type > h6{
    padding-top:0; }

.bp3-dark .bp3-menu{
  background:#30404d;
  color:#f5f8fa; }

.bp3-dark .bp3-menu-item{ }
  .bp3-dark .bp3-menu-item.bp3-intent-primary{
    color:#48aff0; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary::before, .bp3-dark .bp3-menu-item.bp3-intent-primary::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary .bp3-menu-item-label{
      color:#48aff0; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active{
      background-color:#137cbd; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary:active{
      background-color:#106ba3; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-primary:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-primary:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-primary:active, .bp3-dark .bp3-menu-item.bp3-intent-primary:active::before, .bp3-dark .bp3-menu-item.bp3-intent-primary:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item.bp3-intent-success{
    color:#3dcc91; }
    .bp3-dark .bp3-menu-item.bp3-intent-success .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-success::before, .bp3-dark .bp3-menu-item.bp3-intent-success::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success .bp3-menu-item-label{
      color:#3dcc91; }
    .bp3-dark .bp3-menu-item.bp3-intent-success:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active{
      background-color:#0f9960; }
    .bp3-dark .bp3-menu-item.bp3-intent-success:active{
      background-color:#0d8050; }
    .bp3-dark .bp3-menu-item.bp3-intent-success:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-success:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-success:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-success:active, .bp3-dark .bp3-menu-item.bp3-intent-success:active::before, .bp3-dark .bp3-menu-item.bp3-intent-success:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item.bp3-intent-warning{
    color:#ffb366; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning::before, .bp3-dark .bp3-menu-item.bp3-intent-warning::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning .bp3-menu-item-label{
      color:#ffb366; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active{
      background-color:#d9822b; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning:active{
      background-color:#bf7326; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-warning:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-warning:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-warning:active, .bp3-dark .bp3-menu-item.bp3-intent-warning:active::before, .bp3-dark .bp3-menu-item.bp3-intent-warning:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item.bp3-intent-danger{
    color:#ff7373; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger::before, .bp3-dark .bp3-menu-item.bp3-intent-danger::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger .bp3-menu-item-label{
      color:#ff7373; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active{
      background-color:#db3737; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger:active{
      background-color:#c23030; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-danger:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-danger:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-danger:active, .bp3-dark .bp3-menu-item.bp3-intent-danger:active::before, .bp3-dark .bp3-menu-item.bp3-intent-danger:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item::before,
  .bp3-dark .bp3-menu-item > .bp3-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-menu-item .bp3-menu-item-label{
    color:#a7b6c2; }
  .bp3-dark .bp3-menu-item.bp3-active, .bp3-dark .bp3-menu-item:active{
    background-color:rgba(138, 155, 168, 0.3); }
  .bp3-dark .bp3-menu-item.bp3-disabled{
    color:rgba(167, 182, 194, 0.6) !important; }
    .bp3-dark .bp3-menu-item.bp3-disabled::before,
    .bp3-dark .bp3-menu-item.bp3-disabled > .bp3-icon,
    .bp3-dark .bp3-menu-item.bp3-disabled .bp3-menu-item-label{
      color:rgba(167, 182, 194, 0.6) !important; }

.bp3-dark .bp3-menu-divider,
.bp3-dark .bp3-menu-header{
  border-color:rgba(255, 255, 255, 0.15); }

.bp3-dark .bp3-menu-header > h6{
  color:#f5f8fa; }

.bp3-label .bp3-menu{
  margin-top:5px; }
.bp3-navbar{
  background-color:#ffffff;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  height:50px;
  padding:0 15px;
  position:relative;
  width:100%;
  z-index:10; }
  .bp3-navbar.bp3-dark,
  .bp3-dark .bp3-navbar{
    background-color:#394b59; }
  .bp3-navbar.bp3-dark{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-navbar{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-navbar.bp3-fixed-top{
    left:0;
    position:fixed;
    right:0;
    top:0; }

.bp3-navbar-heading{
  font-size:16px;
  margin-right:15px; }

.bp3-navbar-group{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  height:50px; }
  .bp3-navbar-group.bp3-align-left{
    float:left; }
  .bp3-navbar-group.bp3-align-right{
    float:right; }

.bp3-navbar-divider{
  border-left:1px solid rgba(16, 22, 26, 0.15);
  height:20px;
  margin:0 10px; }
  .bp3-dark .bp3-navbar-divider{
    border-left-color:rgba(255, 255, 255, 0.15); }
.bp3-non-ideal-state{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  height:100%;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  text-align:center;
  width:100%; }
  .bp3-non-ideal-state > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-non-ideal-state > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-non-ideal-state::before,
  .bp3-non-ideal-state > *{
    margin-bottom:20px; }
  .bp3-non-ideal-state:empty::before,
  .bp3-non-ideal-state > :last-child{
    margin-bottom:0; }
  .bp3-non-ideal-state > *{
    max-width:400px; }

.bp3-non-ideal-state-visual{
  color:rgba(92, 112, 128, 0.6);
  font-size:60px; }
  .bp3-dark .bp3-non-ideal-state-visual{
    color:rgba(167, 182, 194, 0.6); }

.bp3-overflow-list{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-wrap:nowrap;
      flex-wrap:nowrap;
  min-width:0; }

.bp3-overflow-list-spacer{
  -ms-flex-negative:1;
      flex-shrink:1;
  width:1px; }

body.bp3-overlay-open{
  overflow:hidden; }

.bp3-overlay{
  bottom:0;
  left:0;
  position:static;
  right:0;
  top:0;
  z-index:20; }
  .bp3-overlay:not(.bp3-overlay-open){
    pointer-events:none; }
  .bp3-overlay.bp3-overlay-container{
    overflow:hidden;
    position:fixed; }
    .bp3-overlay.bp3-overlay-container.bp3-overlay-inline{
      position:absolute; }
  .bp3-overlay.bp3-overlay-scroll-container{
    overflow:auto;
    position:fixed; }
    .bp3-overlay.bp3-overlay-scroll-container.bp3-overlay-inline{
      position:absolute; }
  .bp3-overlay.bp3-overlay-inline{
    display:inline;
    overflow:visible; }

.bp3-overlay-content{
  position:fixed;
  z-index:20; }
  .bp3-overlay-inline .bp3-overlay-content,
  .bp3-overlay-scroll-container .bp3-overlay-content{
    position:absolute; }

.bp3-overlay-backdrop{
  bottom:0;
  left:0;
  position:fixed;
  right:0;
  top:0;
  opacity:1;
  background-color:rgba(16, 22, 26, 0.7);
  overflow:auto;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none;
  z-index:20; }
  .bp3-overlay-backdrop.bp3-overlay-enter, .bp3-overlay-backdrop.bp3-overlay-appear{
    opacity:0; }
  .bp3-overlay-backdrop.bp3-overlay-enter-active, .bp3-overlay-backdrop.bp3-overlay-appear-active{
    opacity:1;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-overlay-backdrop.bp3-overlay-exit{
    opacity:1; }
  .bp3-overlay-backdrop.bp3-overlay-exit-active{
    opacity:0;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-overlay-backdrop:focus{
    outline:none; }
  .bp3-overlay-inline .bp3-overlay-backdrop{
    position:absolute; }
.bp3-panel-stack{
  overflow:hidden;
  position:relative; }

.bp3-panel-stack-header{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-shadow:0 1px rgba(16, 22, 26, 0.15);
          box-shadow:0 1px rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-negative:0;
      flex-shrink:0;
  height:30px;
  z-index:1; }
  .bp3-dark .bp3-panel-stack-header{
    -webkit-box-shadow:0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 1px rgba(255, 255, 255, 0.15); }
  .bp3-panel-stack-header > span{
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1;
            flex:1; }
  .bp3-panel-stack-header .bp3-heading{
    margin:0 5px; }

.bp3-button.bp3-panel-stack-header-back{
  margin-left:5px;
  padding-left:0;
  white-space:nowrap; }
  .bp3-button.bp3-panel-stack-header-back .bp3-icon{
    margin:0 2px; }

.bp3-panel-stack-view{
  bottom:0;
  left:0;
  position:absolute;
  right:0;
  top:0;
  background-color:#ffffff;
  border-right:1px solid rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin-right:-1px;
  overflow-y:auto;
  z-index:1; }
  .bp3-dark .bp3-panel-stack-view{
    background-color:#30404d; }
  .bp3-panel-stack-view:nth-last-child(n + 4){
    display:none; }

.bp3-panel-stack-push .bp3-panel-stack-enter, .bp3-panel-stack-push .bp3-panel-stack-appear{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0; }

.bp3-panel-stack-push .bp3-panel-stack-enter-active, .bp3-panel-stack-push .bp3-panel-stack-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack-push .bp3-panel-stack-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack-push .bp3-panel-stack-exit-active{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack-pop .bp3-panel-stack-enter, .bp3-panel-stack-pop .bp3-panel-stack-appear{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0; }

.bp3-panel-stack-pop .bp3-panel-stack-enter-active, .bp3-panel-stack-pop .bp3-panel-stack-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack-pop .bp3-panel-stack-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack-pop .bp3-panel-stack-exit-active{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }
.bp3-popover{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  -webkit-transform:scale(1);
          transform:scale(1);
  border-radius:3px;
  display:inline-block;
  z-index:20; }
  .bp3-popover .bp3-popover-arrow{
    height:30px;
    position:absolute;
    width:30px; }
    .bp3-popover .bp3-popover-arrow::before{
      height:20px;
      margin:5px;
      width:20px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover{
    margin-bottom:17px;
    margin-top:-17px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow{
      bottom:-11px; }
      .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(-90deg);
                transform:rotate(-90deg); }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover{
    margin-left:17px; }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow{
      left:-11px; }
      .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(0);
                transform:rotate(0); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover{
    margin-top:17px; }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow{
      top:-11px; }
      .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(90deg);
                transform:rotate(90deg); }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover{
    margin-left:-17px;
    margin-right:17px; }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow{
      right:-11px; }
      .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(180deg);
                transform:rotate(180deg); }
  .bp3-tether-element-attached-middle > .bp3-popover > .bp3-popover-arrow{
    top:50%;
    -webkit-transform:translateY(-50%);
            transform:translateY(-50%); }
  .bp3-tether-element-attached-center > .bp3-popover > .bp3-popover-arrow{
    right:50%;
    -webkit-transform:translateX(50%);
            transform:translateX(50%); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow{
    top:-0.3934px; }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow{
    right:-0.3934px; }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow{
    left:-0.3934px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow{
    bottom:-0.3934px; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:top left;
            transform-origin:top left; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:top center;
            transform-origin:top center; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:top right;
            transform-origin:top right; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:center left;
            transform-origin:center left; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:center center;
            transform-origin:center center; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:center right;
            transform-origin:center right; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:bottom left;
            transform-origin:bottom left; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:bottom center;
            transform-origin:bottom center; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:bottom right;
            transform-origin:bottom right; }
  .bp3-popover .bp3-popover-content{
    background:#ffffff;
    color:inherit; }
  .bp3-popover .bp3-popover-arrow::before{
    -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2);
            box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2); }
  .bp3-popover .bp3-popover-arrow-border{
    fill:#10161a;
    fill-opacity:0.1; }
  .bp3-popover .bp3-popover-arrow-fill{
    fill:#ffffff; }
  .bp3-popover-enter > .bp3-popover, .bp3-popover-appear > .bp3-popover{
    -webkit-transform:scale(0.3);
            transform:scale(0.3); }
  .bp3-popover-enter-active > .bp3-popover, .bp3-popover-appear-active > .bp3-popover{
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-popover-exit > .bp3-popover{
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-popover-exit-active > .bp3-popover{
    -webkit-transform:scale(0.3);
            transform:scale(0.3);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-popover .bp3-popover-content{
    border-radius:3px;
    position:relative; }
  .bp3-popover.bp3-popover-content-sizing .bp3-popover-content{
    max-width:350px;
    padding:20px; }
  .bp3-popover-target + .bp3-overlay .bp3-popover.bp3-popover-content-sizing{
    width:350px; }
  .bp3-popover.bp3-minimal{
    margin:0 !important; }
    .bp3-popover.bp3-minimal .bp3-popover-arrow{
      display:none; }
    .bp3-popover.bp3-minimal.bp3-popover{
      -webkit-transform:scale(1);
              transform:scale(1); }
      .bp3-popover-enter > .bp3-popover.bp3-minimal.bp3-popover, .bp3-popover-appear > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1); }
      .bp3-popover-enter-active > .bp3-popover.bp3-minimal.bp3-popover, .bp3-popover-appear-active > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
      .bp3-popover-exit > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1); }
      .bp3-popover-exit-active > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-popover.bp3-dark,
  .bp3-dark .bp3-popover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-popover.bp3-dark .bp3-popover-content,
    .bp3-dark .bp3-popover .bp3-popover-content{
      background:#30404d;
      color:inherit; }
    .bp3-popover.bp3-dark .bp3-popover-arrow::before,
    .bp3-dark .bp3-popover .bp3-popover-arrow::before{
      -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4);
              box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4); }
    .bp3-popover.bp3-dark .bp3-popover-arrow-border,
    .bp3-dark .bp3-popover .bp3-popover-arrow-border{
      fill:#10161a;
      fill-opacity:0.2; }
    .bp3-popover.bp3-dark .bp3-popover-arrow-fill,
    .bp3-dark .bp3-popover .bp3-popover-arrow-fill{
      fill:#30404d; }

.bp3-popover-arrow::before{
  border-radius:2px;
  content:"";
  display:block;
  position:absolute;
  -webkit-transform:rotate(45deg);
          transform:rotate(45deg); }

.bp3-tether-pinned .bp3-popover-arrow{
  display:none; }

.bp3-popover-backdrop{
  background:rgba(255, 255, 255, 0); }

.bp3-transition-container{
  opacity:1;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  z-index:20; }
  .bp3-transition-container.bp3-popover-enter, .bp3-transition-container.bp3-popover-appear{
    opacity:0; }
  .bp3-transition-container.bp3-popover-enter-active, .bp3-transition-container.bp3-popover-appear-active{
    opacity:1;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-transition-container.bp3-popover-exit{
    opacity:1; }
  .bp3-transition-container.bp3-popover-exit-active{
    opacity:0;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-transition-container:focus{
    outline:none; }
  .bp3-transition-container.bp3-popover-leave .bp3-popover-content{
    pointer-events:none; }
  .bp3-transition-container[data-x-out-of-boundaries]{
    display:none; }

span.bp3-popover-target{
  display:inline-block; }

.bp3-popover-wrapper.bp3-fill{
  width:100%; }

.bp3-portal{
  left:0;
  position:absolute;
  right:0;
  top:0; }
@-webkit-keyframes linear-progress-bar-stripes{
  from{
    background-position:0 0; }
  to{
    background-position:30px 0; } }
@keyframes linear-progress-bar-stripes{
  from{
    background-position:0 0; }
  to{
    background-position:30px 0; } }

.bp3-progress-bar{
  background:rgba(92, 112, 128, 0.2);
  border-radius:40px;
  display:block;
  height:8px;
  overflow:hidden;
  position:relative;
  width:100%; }
  .bp3-progress-bar .bp3-progress-meter{
    background:linear-gradient(-45deg, rgba(255, 255, 255, 0.2) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.2) 50%, rgba(255, 255, 255, 0.2) 75%, transparent 75%);
    background-color:rgba(92, 112, 128, 0.8);
    background-size:30px 30px;
    border-radius:40px;
    height:100%;
    position:absolute;
    -webkit-transition:width 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:width 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    width:100%; }
  .bp3-progress-bar:not(.bp3-no-animation):not(.bp3-no-stripes) .bp3-progress-meter{
    animation:linear-progress-bar-stripes 300ms linear infinite reverse; }
  .bp3-progress-bar.bp3-no-stripes .bp3-progress-meter{
    background-image:none; }

.bp3-dark .bp3-progress-bar{
  background:rgba(16, 22, 26, 0.5); }
  .bp3-dark .bp3-progress-bar .bp3-progress-meter{
    background-color:#8a9ba8; }

.bp3-progress-bar.bp3-intent-primary .bp3-progress-meter{
  background-color:#137cbd; }

.bp3-progress-bar.bp3-intent-success .bp3-progress-meter{
  background-color:#0f9960; }

.bp3-progress-bar.bp3-intent-warning .bp3-progress-meter{
  background-color:#d9822b; }

.bp3-progress-bar.bp3-intent-danger .bp3-progress-meter{
  background-color:#db3737; }
@-webkit-keyframes skeleton-glow{
  from{
    background:rgba(206, 217, 224, 0.2);
    border-color:rgba(206, 217, 224, 0.2); }
  to{
    background:rgba(92, 112, 128, 0.2);
    border-color:rgba(92, 112, 128, 0.2); } }
@keyframes skeleton-glow{
  from{
    background:rgba(206, 217, 224, 0.2);
    border-color:rgba(206, 217, 224, 0.2); }
  to{
    background:rgba(92, 112, 128, 0.2);
    border-color:rgba(92, 112, 128, 0.2); } }
.bp3-skeleton{
  -webkit-animation:1000ms linear infinite alternate skeleton-glow;
          animation:1000ms linear infinite alternate skeleton-glow;
  background:rgba(206, 217, 224, 0.2);
  background-clip:padding-box !important;
  border-color:rgba(206, 217, 224, 0.2) !important;
  border-radius:2px;
  -webkit-box-shadow:none !important;
          box-shadow:none !important;
  color:transparent !important;
  cursor:default;
  pointer-events:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-skeleton::before, .bp3-skeleton::after,
  .bp3-skeleton *{
    visibility:hidden !important; }
.bp3-slider{
  height:40px;
  min-width:150px;
  width:100%;
  cursor:default;
  outline:none;
  position:relative;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-slider:hover{
    cursor:pointer; }
  .bp3-slider:active{
    cursor:-webkit-grabbing;
    cursor:grabbing; }
  .bp3-slider.bp3-disabled{
    cursor:not-allowed;
    opacity:0.5; }
  .bp3-slider.bp3-slider-unlabeled{
    height:16px; }

.bp3-slider-track,
.bp3-slider-progress{
  height:6px;
  left:0;
  right:0;
  top:5px;
  position:absolute; }

.bp3-slider-track{
  border-radius:3px;
  overflow:hidden; }

.bp3-slider-progress{
  background:rgba(92, 112, 128, 0.2); }
  .bp3-dark .bp3-slider-progress{
    background:rgba(16, 22, 26, 0.5); }
  .bp3-slider-progress.bp3-intent-primary{
    background-color:#137cbd; }
  .bp3-slider-progress.bp3-intent-success{
    background-color:#0f9960; }
  .bp3-slider-progress.bp3-intent-warning{
    background-color:#d9822b; }
  .bp3-slider-progress.bp3-intent-danger{
    background-color:#db3737; }

.bp3-slider-handle{
  background-color:#f5f8fa;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
  color:#182026;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
  cursor:pointer;
  height:16px;
  left:0;
  position:absolute;
  top:0;
  width:16px; }
  .bp3-slider-handle:hover{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
  .bp3-slider-handle:active, .bp3-slider-handle.bp3-active{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-slider-handle:disabled, .bp3-slider-handle.bp3-disabled{
    background-color:rgba(206, 217, 224, 0.5);
    background-image:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    outline:none; }
    .bp3-slider-handle:disabled.bp3-active, .bp3-slider-handle:disabled.bp3-active:hover, .bp3-slider-handle.bp3-disabled.bp3-active, .bp3-slider-handle.bp3-disabled.bp3-active:hover{
      background:rgba(206, 217, 224, 0.7); }
  .bp3-slider-handle:focus{
    z-index:1; }
  .bp3-slider-handle:hover{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
    cursor:-webkit-grab;
    cursor:grab;
    z-index:2; }
  .bp3-slider-handle.bp3-active{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 1px rgba(16, 22, 26, 0.1);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 1px rgba(16, 22, 26, 0.1);
    cursor:-webkit-grabbing;
    cursor:grabbing; }
  .bp3-disabled .bp3-slider-handle{
    background:#bfccd6;
    -webkit-box-shadow:none;
            box-shadow:none;
    pointer-events:none; }
  .bp3-dark .bp3-slider-handle{
    background-color:#394b59;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark .bp3-slider-handle:hover, .bp3-dark .bp3-slider-handle:active, .bp3-dark .bp3-slider-handle.bp3-active{
      color:#f5f8fa; }
    .bp3-dark .bp3-slider-handle:hover{
      background-color:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-slider-handle:active, .bp3-dark .bp3-slider-handle.bp3-active{
      background-color:#202b33;
      background-image:none;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-slider-handle:disabled, .bp3-dark .bp3-slider-handle.bp3-disabled{
      background-color:rgba(57, 75, 89, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-slider-handle:disabled.bp3-active, .bp3-dark .bp3-slider-handle.bp3-disabled.bp3-active{
        background:rgba(57, 75, 89, 0.7); }
    .bp3-dark .bp3-slider-handle .bp3-button-spinner .bp3-spinner-head{
      background:rgba(16, 22, 26, 0.5);
      stroke:#8a9ba8; }
    .bp3-dark .bp3-slider-handle, .bp3-dark .bp3-slider-handle:hover{
      background-color:#394b59; }
    .bp3-dark .bp3-slider-handle.bp3-active{
      background-color:#293742; }
  .bp3-dark .bp3-disabled .bp3-slider-handle{
    background:#5c7080;
    border-color:#5c7080;
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-slider-handle .bp3-slider-label{
    background:#394b59;
    border-radius:3px;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
    color:#f5f8fa;
    margin-left:8px; }
    .bp3-dark .bp3-slider-handle .bp3-slider-label{
      background:#e1e8ed;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
      color:#394b59; }
    .bp3-disabled .bp3-slider-handle .bp3-slider-label{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-slider-handle.bp3-start, .bp3-slider-handle.bp3-end{
    width:8px; }
  .bp3-slider-handle.bp3-start{
    border-bottom-right-radius:0;
    border-top-right-radius:0; }
  .bp3-slider-handle.bp3-end{
    border-bottom-left-radius:0;
    border-top-left-radius:0;
    margin-left:8px; }
    .bp3-slider-handle.bp3-end .bp3-slider-label{
      margin-left:0; }

.bp3-slider-label{
  -webkit-transform:translate(-50%, 20px);
          transform:translate(-50%, 20px);
  display:inline-block;
  font-size:12px;
  line-height:1;
  padding:2px 5px;
  position:absolute;
  vertical-align:top; }

.bp3-slider.bp3-vertical{
  height:150px;
  min-width:40px;
  width:40px; }
  .bp3-slider.bp3-vertical .bp3-slider-track,
  .bp3-slider.bp3-vertical .bp3-slider-progress{
    bottom:0;
    height:auto;
    left:5px;
    top:0;
    width:6px; }
  .bp3-slider.bp3-vertical .bp3-slider-progress{
    top:auto; }
  .bp3-slider.bp3-vertical .bp3-slider-label{
    -webkit-transform:translate(20px, 50%);
            transform:translate(20px, 50%); }
  .bp3-slider.bp3-vertical .bp3-slider-handle{
    top:auto; }
    .bp3-slider.bp3-vertical .bp3-slider-handle .bp3-slider-label{
      margin-left:0;
      margin-top:-8px; }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-end, .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start{
      height:8px;
      margin-left:0;
      width:16px; }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start{
      border-bottom-right-radius:3px;
      border-top-left-radius:0; }
      .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start .bp3-slider-label{
        -webkit-transform:translate(20px);
                transform:translate(20px); }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-end{
      border-bottom-left-radius:0;
      border-bottom-right-radius:0;
      border-top-left-radius:3px;
      margin-bottom:8px; }

@-webkit-keyframes pt-spinner-animation{
  from{
    -webkit-transform:rotate(0deg);
            transform:rotate(0deg); }
  to{
    -webkit-transform:rotate(360deg);
            transform:rotate(360deg); } }

@keyframes pt-spinner-animation{
  from{
    -webkit-transform:rotate(0deg);
            transform:rotate(0deg); }
  to{
    -webkit-transform:rotate(360deg);
            transform:rotate(360deg); } }

.bp3-spinner{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  overflow:visible;
  vertical-align:middle; }
  .bp3-spinner svg{
    display:block; }
  .bp3-spinner path{
    fill-opacity:0; }
  .bp3-spinner .bp3-spinner-head{
    stroke:rgba(92, 112, 128, 0.8);
    stroke-linecap:round;
    -webkit-transform-origin:center;
            transform-origin:center;
    -webkit-transition:stroke-dashoffset 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:stroke-dashoffset 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-spinner .bp3-spinner-track{
    stroke:rgba(92, 112, 128, 0.2); }

.bp3-spinner-animation{
  -webkit-animation:pt-spinner-animation 500ms linear infinite;
          animation:pt-spinner-animation 500ms linear infinite; }
  .bp3-no-spin > .bp3-spinner-animation{
    -webkit-animation:none;
            animation:none; }

.bp3-dark .bp3-spinner .bp3-spinner-head{
  stroke:#8a9ba8; }

.bp3-dark .bp3-spinner .bp3-spinner-track{
  stroke:rgba(16, 22, 26, 0.5); }

.bp3-spinner.bp3-intent-primary .bp3-spinner-head{
  stroke:#137cbd; }

.bp3-spinner.bp3-intent-success .bp3-spinner-head{
  stroke:#0f9960; }

.bp3-spinner.bp3-intent-warning .bp3-spinner-head{
  stroke:#d9822b; }

.bp3-spinner.bp3-intent-danger .bp3-spinner-head{
  stroke:#db3737; }
.bp3-tabs.bp3-vertical{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }
  .bp3-tabs.bp3-vertical > .bp3-tab-list{
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start;
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column; }
    .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab{
      border-radius:3px;
      padding:0 10px;
      width:100%; }
      .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab[aria-selected="true"]{
        background-color:rgba(19, 124, 189, 0.2);
        -webkit-box-shadow:none;
                box-shadow:none; }
    .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab-indicator-wrapper .bp3-tab-indicator{
      background-color:rgba(19, 124, 189, 0.2);
      border-radius:3px;
      bottom:0;
      height:auto;
      left:0;
      right:0;
      top:0; }
  .bp3-tabs.bp3-vertical > .bp3-tab-panel{
    margin-top:0;
    padding-left:20px; }

.bp3-tab-list{
  -webkit-box-align:end;
      -ms-flex-align:end;
          align-items:flex-end;
  border:none;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  list-style:none;
  margin:0;
  padding:0;
  position:relative; }
  .bp3-tab-list > *:not(:last-child){
    margin-right:20px; }

.bp3-tab{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  color:#182026;
  cursor:pointer;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  font-size:14px;
  line-height:30px;
  max-width:100%;
  position:relative;
  vertical-align:top; }
  .bp3-tab a{
    color:inherit;
    display:block;
    text-decoration:none; }
  .bp3-tab-indicator-wrapper ~ .bp3-tab{
    background-color:transparent !important;
    -webkit-box-shadow:none !important;
            box-shadow:none !important; }
  .bp3-tab[aria-disabled="true"]{
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  .bp3-tab[aria-selected="true"]{
    border-radius:0;
    -webkit-box-shadow:inset 0 -3px 0 #106ba3;
            box-shadow:inset 0 -3px 0 #106ba3; }
  .bp3-tab[aria-selected="true"], .bp3-tab:not([aria-disabled="true"]):hover{
    color:#106ba3; }
  .bp3-tab:focus{
    -moz-outline-radius:0; }
  .bp3-large > .bp3-tab{
    font-size:16px;
    line-height:40px; }

.bp3-tab-panel{
  margin-top:20px; }
  .bp3-tab-panel[aria-hidden="true"]{
    display:none; }

.bp3-tab-indicator-wrapper{
  left:0;
  pointer-events:none;
  position:absolute;
  top:0;
  -webkit-transform:translateX(0), translateY(0);
          transform:translateX(0), translateY(0);
  -webkit-transition:height, width, -webkit-transform;
  transition:height, width, -webkit-transform;
  transition:height, transform, width;
  transition:height, transform, width, -webkit-transform;
  -webkit-transition-duration:200ms;
          transition-duration:200ms;
  -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
          transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-tab-indicator-wrapper .bp3-tab-indicator{
    background-color:#106ba3;
    bottom:0;
    height:3px;
    left:0;
    position:absolute;
    right:0; }
  .bp3-tab-indicator-wrapper.bp3-no-animation{
    -webkit-transition:none;
    transition:none; }

.bp3-dark .bp3-tab{
  color:#f5f8fa; }
  .bp3-dark .bp3-tab[aria-disabled="true"]{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-tab[aria-selected="true"]{
    -webkit-box-shadow:inset 0 -3px 0 #48aff0;
            box-shadow:inset 0 -3px 0 #48aff0; }
  .bp3-dark .bp3-tab[aria-selected="true"], .bp3-dark .bp3-tab:not([aria-disabled="true"]):hover{
    color:#48aff0; }

.bp3-dark .bp3-tab-indicator{
  background-color:#48aff0; }

.bp3-flex-expander{
  -webkit-box-flex:1;
      -ms-flex:1 1;
          flex:1 1; }
.bp3-tag{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background-color:#5c7080;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:none;
          box-shadow:none;
  color:#f5f8fa;
  font-size:12px;
  line-height:16px;
  max-width:100%;
  min-height:20px;
  min-width:20px;
  padding:2px 6px;
  position:relative; }
  .bp3-tag.bp3-interactive{
    cursor:pointer; }
    .bp3-tag.bp3-interactive:hover{
      background-color:rgba(92, 112, 128, 0.85); }
    .bp3-tag.bp3-interactive.bp3-active, .bp3-tag.bp3-interactive:active{
      background-color:rgba(92, 112, 128, 0.7); }
  .bp3-tag > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-tag > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-tag::before,
  .bp3-tag > *{
    margin-right:4px; }
  .bp3-tag:empty::before,
  .bp3-tag > :last-child{
    margin-right:0; }
  .bp3-tag:focus{
    outline:rgba(19, 124, 189, 0.6) auto 2px;
    outline-offset:0;
    -moz-outline-radius:6px; }
  .bp3-tag.bp3-round{
    border-radius:30px;
    padding-left:8px;
    padding-right:8px; }
  .bp3-dark .bp3-tag{
    background-color:#bfccd6;
    color:#182026; }
    .bp3-dark .bp3-tag.bp3-interactive{
      cursor:pointer; }
      .bp3-dark .bp3-tag.bp3-interactive:hover{
        background-color:rgba(191, 204, 214, 0.85); }
      .bp3-dark .bp3-tag.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-interactive:active{
        background-color:rgba(191, 204, 214, 0.7); }
    .bp3-dark .bp3-tag > .bp3-icon, .bp3-dark .bp3-tag .bp3-icon-standard, .bp3-dark .bp3-tag .bp3-icon-large{
      fill:currentColor; }
  .bp3-tag > .bp3-icon, .bp3-tag .bp3-icon-standard, .bp3-tag .bp3-icon-large{
    fill:#ffffff; }
  .bp3-tag.bp3-large,
  .bp3-large .bp3-tag{
    font-size:14px;
    line-height:20px;
    min-height:30px;
    min-width:30px;
    padding:5px 10px; }
    .bp3-tag.bp3-large::before,
    .bp3-tag.bp3-large > *,
    .bp3-large .bp3-tag::before,
    .bp3-large .bp3-tag > *{
      margin-right:7px; }
    .bp3-tag.bp3-large:empty::before,
    .bp3-tag.bp3-large > :last-child,
    .bp3-large .bp3-tag:empty::before,
    .bp3-large .bp3-tag > :last-child{
      margin-right:0; }
    .bp3-tag.bp3-large.bp3-round,
    .bp3-large .bp3-tag.bp3-round{
      padding-left:12px;
      padding-right:12px; }
  .bp3-tag.bp3-intent-primary{
    background:#137cbd;
    color:#ffffff; }
    .bp3-tag.bp3-intent-primary.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-primary.bp3-interactive:hover{
        background-color:rgba(19, 124, 189, 0.85); }
      .bp3-tag.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-primary.bp3-interactive:active{
        background-color:rgba(19, 124, 189, 0.7); }
  .bp3-tag.bp3-intent-success{
    background:#0f9960;
    color:#ffffff; }
    .bp3-tag.bp3-intent-success.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-success.bp3-interactive:hover{
        background-color:rgba(15, 153, 96, 0.85); }
      .bp3-tag.bp3-intent-success.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-success.bp3-interactive:active{
        background-color:rgba(15, 153, 96, 0.7); }
  .bp3-tag.bp3-intent-warning{
    background:#d9822b;
    color:#ffffff; }
    .bp3-tag.bp3-intent-warning.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-warning.bp3-interactive:hover{
        background-color:rgba(217, 130, 43, 0.85); }
      .bp3-tag.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-warning.bp3-interactive:active{
        background-color:rgba(217, 130, 43, 0.7); }
  .bp3-tag.bp3-intent-danger{
    background:#db3737;
    color:#ffffff; }
    .bp3-tag.bp3-intent-danger.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-danger.bp3-interactive:hover{
        background-color:rgba(219, 55, 55, 0.85); }
      .bp3-tag.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-danger.bp3-interactive:active{
        background-color:rgba(219, 55, 55, 0.7); }
  .bp3-tag.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-tag.bp3-minimal > .bp3-icon, .bp3-tag.bp3-minimal .bp3-icon-standard, .bp3-tag.bp3-minimal .bp3-icon-large{
    fill:#5c7080; }
  .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]){
    background-color:rgba(138, 155, 168, 0.2);
    color:#182026; }
    .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:hover{
        background-color:rgba(92, 112, 128, 0.3); }
      .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive.bp3-active, .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:active{
        background-color:rgba(92, 112, 128, 0.4); }
    .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]){
      color:#f5f8fa; }
      .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:hover{
          background-color:rgba(191, 204, 214, 0.3); }
        .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:active{
          background-color:rgba(191, 204, 214, 0.4); }
      .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) > .bp3-icon, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) .bp3-icon-standard, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) .bp3-icon-large{
        fill:#a7b6c2; }
  .bp3-tag.bp3-minimal.bp3-intent-primary{
    background-color:rgba(19, 124, 189, 0.15);
    color:#106ba3; }
    .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:hover{
        background-color:rgba(19, 124, 189, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:active{
        background-color:rgba(19, 124, 189, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-primary > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-primary .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-primary .bp3-icon-large{
      fill:#137cbd; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary{
      background-color:rgba(19, 124, 189, 0.25);
      color:#48aff0; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:hover{
          background-color:rgba(19, 124, 189, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:active{
          background-color:rgba(19, 124, 189, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-success{
    background-color:rgba(15, 153, 96, 0.15);
    color:#0d8050; }
    .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:hover{
        background-color:rgba(15, 153, 96, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:active{
        background-color:rgba(15, 153, 96, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-success > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-success .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-success .bp3-icon-large{
      fill:#0f9960; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success{
      background-color:rgba(15, 153, 96, 0.25);
      color:#3dcc91; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:hover{
          background-color:rgba(15, 153, 96, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:active{
          background-color:rgba(15, 153, 96, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-warning{
    background-color:rgba(217, 130, 43, 0.15);
    color:#bf7326; }
    .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:hover{
        background-color:rgba(217, 130, 43, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:active{
        background-color:rgba(217, 130, 43, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-warning > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-warning .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-warning .bp3-icon-large{
      fill:#d9822b; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning{
      background-color:rgba(217, 130, 43, 0.25);
      color:#ffb366; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:hover{
          background-color:rgba(217, 130, 43, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:active{
          background-color:rgba(217, 130, 43, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-danger{
    background-color:rgba(219, 55, 55, 0.15);
    color:#c23030; }
    .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:hover{
        background-color:rgba(219, 55, 55, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:active{
        background-color:rgba(219, 55, 55, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-danger > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-danger .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-danger .bp3-icon-large{
      fill:#db3737; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger{
      background-color:rgba(219, 55, 55, 0.25);
      color:#ff7373; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:hover{
          background-color:rgba(219, 55, 55, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:active{
          background-color:rgba(219, 55, 55, 0.45); }

.bp3-tag-remove{
  background:none;
  border:none;
  color:inherit;
  cursor:pointer;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  margin-bottom:-2px;
  margin-right:-6px !important;
  margin-top:-2px;
  opacity:0.5;
  padding:2px;
  padding-left:0; }
  .bp3-tag-remove:hover{
    background:none;
    opacity:0.8;
    text-decoration:none; }
  .bp3-tag-remove:active{
    opacity:1; }
  .bp3-tag-remove:empty::before{
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-style:normal;
    font-weight:400;
    line-height:1;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    content:""; }
  .bp3-large .bp3-tag-remove{
    margin-right:-10px !important;
    padding:0 5px 0 0; }
    .bp3-large .bp3-tag-remove:empty::before{
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-style:normal;
      font-weight:400;
      line-height:1; }
.bp3-tag-input{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  cursor:text;
  height:auto;
  line-height:inherit;
  min-height:30px;
  padding-left:5px;
  padding-right:0; }
  .bp3-tag-input > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-tag-input > .bp3-tag-input-values{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-tag-input .bp3-tag-input-icon{
    color:#5c7080;
    margin-left:2px;
    margin-right:7px;
    margin-top:7px; }
  .bp3-tag-input .bp3-tag-input-values{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row;
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    -ms-flex-item-align:stretch;
        align-self:stretch;
    -ms-flex-wrap:wrap;
        flex-wrap:wrap;
    margin-right:7px;
    margin-top:5px;
    min-width:0; }
    .bp3-tag-input .bp3-tag-input-values > *{
      -webkit-box-flex:0;
          -ms-flex-positive:0;
              flex-grow:0;
      -ms-flex-negative:0;
          flex-shrink:0; }
    .bp3-tag-input .bp3-tag-input-values > .bp3-fill{
      -webkit-box-flex:1;
          -ms-flex-positive:1;
              flex-grow:1;
      -ms-flex-negative:1;
          flex-shrink:1; }
    .bp3-tag-input .bp3-tag-input-values::before,
    .bp3-tag-input .bp3-tag-input-values > *{
      margin-right:5px; }
    .bp3-tag-input .bp3-tag-input-values:empty::before,
    .bp3-tag-input .bp3-tag-input-values > :last-child{
      margin-right:0; }
    .bp3-tag-input .bp3-tag-input-values:first-child .bp3-input-ghost:first-child{
      padding-left:5px; }
    .bp3-tag-input .bp3-tag-input-values > *{
      margin-bottom:5px; }
  .bp3-tag-input .bp3-tag{
    overflow-wrap:break-word; }
    .bp3-tag-input .bp3-tag.bp3-active{
      outline:rgba(19, 124, 189, 0.6) auto 2px;
      outline-offset:0;
      -moz-outline-radius:6px; }
  .bp3-tag-input .bp3-input-ghost{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    line-height:20px;
    width:80px; }
    .bp3-tag-input .bp3-input-ghost:disabled, .bp3-tag-input .bp3-input-ghost.bp3-disabled{
      cursor:not-allowed; }
  .bp3-tag-input .bp3-button,
  .bp3-tag-input .bp3-spinner{
    margin:3px;
    margin-left:0; }
  .bp3-tag-input .bp3-button{
    min-height:24px;
    min-width:24px;
    padding:0 7px; }
  .bp3-tag-input.bp3-large{
    height:auto;
    min-height:40px; }
    .bp3-tag-input.bp3-large::before,
    .bp3-tag-input.bp3-large > *{
      margin-right:10px; }
    .bp3-tag-input.bp3-large:empty::before,
    .bp3-tag-input.bp3-large > :last-child{
      margin-right:0; }
    .bp3-tag-input.bp3-large .bp3-tag-input-icon{
      margin-left:5px;
      margin-top:10px; }
    .bp3-tag-input.bp3-large .bp3-input-ghost{
      line-height:30px; }
    .bp3-tag-input.bp3-large .bp3-button{
      min-height:30px;
      min-width:30px;
      padding:5px 10px;
      margin:5px;
      margin-left:0; }
    .bp3-tag-input.bp3-large .bp3-spinner{
      margin:8px;
      margin-left:0; }
  .bp3-tag-input.bp3-active{
    background-color:#ffffff;
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-success{
      -webkit-box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-tag-input .bp3-tag-input-icon, .bp3-tag-input.bp3-dark .bp3-tag-input-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-tag-input .bp3-input-ghost, .bp3-tag-input.bp3-dark .bp3-input-ghost{
    color:#f5f8fa; }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-webkit-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-moz-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost:-ms-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-ms-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::placeholder{
      color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-tag-input.bp3-active, .bp3-tag-input.bp3-dark.bp3-active{
    background-color:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-primary, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-success, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-success{
      -webkit-box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-warning, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-danger, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-input-ghost{
  background:none;
  border:none;
  -webkit-box-shadow:none;
          box-shadow:none;
  padding:0; }
  .bp3-input-ghost::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost:focus{
    outline:none !important; }
.bp3-toast{
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  background-color:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  margin:20px 0 0;
  max-width:500px;
  min-width:300px;
  pointer-events:all;
  position:relative !important; }
  .bp3-toast.bp3-toast-enter, .bp3-toast.bp3-toast-appear{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px); }
  .bp3-toast.bp3-toast-enter-active, .bp3-toast.bp3-toast-appear-active{
    -webkit-transform:translateY(0);
            transform:translateY(0);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-toast.bp3-toast-enter ~ .bp3-toast, .bp3-toast.bp3-toast-appear ~ .bp3-toast{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px); }
  .bp3-toast.bp3-toast-enter-active ~ .bp3-toast, .bp3-toast.bp3-toast-appear-active ~ .bp3-toast{
    -webkit-transform:translateY(0);
            transform:translateY(0);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-toast.bp3-toast-exit{
    opacity:1;
    -webkit-filter:blur(0);
            filter:blur(0); }
  .bp3-toast.bp3-toast-exit-active{
    opacity:0;
    -webkit-filter:blur(10px);
            filter:blur(10px);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:opacity, filter;
    transition-property:opacity, filter, -webkit-filter;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-toast.bp3-toast-exit ~ .bp3-toast{
    -webkit-transform:translateY(0);
            transform:translateY(0); }
  .bp3-toast.bp3-toast-exit-active ~ .bp3-toast{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px);
    -webkit-transition-delay:50ms;
            transition-delay:50ms;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-toast .bp3-button-group{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    padding:5px;
    padding-left:0; }
  .bp3-toast > .bp3-icon{
    color:#5c7080;
    margin:12px;
    margin-right:0; }
  .bp3-toast.bp3-dark,
  .bp3-dark .bp3-toast{
    background-color:#394b59;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-toast.bp3-dark > .bp3-icon,
    .bp3-dark .bp3-toast > .bp3-icon{
      color:#a7b6c2; }
  .bp3-toast[class*="bp3-intent-"] a{
    color:rgba(255, 255, 255, 0.7); }
    .bp3-toast[class*="bp3-intent-"] a:hover{
      color:#ffffff; }
  .bp3-toast[class*="bp3-intent-"] > .bp3-icon{
    color:#ffffff; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button, .bp3-toast[class*="bp3-intent-"] .bp3-button::before,
  .bp3-toast[class*="bp3-intent-"] .bp3-button .bp3-icon, .bp3-toast[class*="bp3-intent-"] .bp3-button:active{
    color:rgba(255, 255, 255, 0.7) !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:focus{
    outline-color:rgba(255, 255, 255, 0.5); }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:hover{
    background-color:rgba(255, 255, 255, 0.15) !important;
    color:#ffffff !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:active{
    background-color:rgba(255, 255, 255, 0.3) !important;
    color:#ffffff !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button::after{
    background:rgba(255, 255, 255, 0.3) !important; }
  .bp3-toast.bp3-intent-primary{
    background-color:#137cbd;
    color:#ffffff; }
  .bp3-toast.bp3-intent-success{
    background-color:#0f9960;
    color:#ffffff; }
  .bp3-toast.bp3-intent-warning{
    background-color:#d9822b;
    color:#ffffff; }
  .bp3-toast.bp3-intent-danger{
    background-color:#db3737;
    color:#ffffff; }

.bp3-toast-message{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  padding:11px;
  word-break:break-word; }

.bp3-toast-container{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box !important;
  display:-ms-flexbox !important;
  display:flex !important;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  left:0;
  overflow:hidden;
  padding:0 20px 20px;
  pointer-events:none;
  position:fixed;
  right:0;
  z-index:40; }
  .bp3-toast-container.bp3-toast-container-top{
    top:0; }
  .bp3-toast-container.bp3-toast-container-bottom{
    bottom:0;
    -webkit-box-orient:vertical;
    -webkit-box-direction:reverse;
        -ms-flex-direction:column-reverse;
            flex-direction:column-reverse;
    top:auto; }
  .bp3-toast-container.bp3-toast-container-left{
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start; }
  .bp3-toast-container.bp3-toast-container-right{
    -webkit-box-align:end;
        -ms-flex-align:end;
            align-items:flex-end; }

.bp3-toast-container-bottom .bp3-toast.bp3-toast-enter:not(.bp3-toast-enter-active),
.bp3-toast-container-bottom .bp3-toast.bp3-toast-enter:not(.bp3-toast-enter-active) ~ .bp3-toast, .bp3-toast-container-bottom .bp3-toast.bp3-toast-appear:not(.bp3-toast-appear-active),
.bp3-toast-container-bottom .bp3-toast.bp3-toast-appear:not(.bp3-toast-appear-active) ~ .bp3-toast,
.bp3-toast-container-bottom .bp3-toast.bp3-toast-exit-active ~ .bp3-toast,
.bp3-toast-container-bottom .bp3-toast.bp3-toast-leave-active ~ .bp3-toast{
  -webkit-transform:translateY(60px);
          transform:translateY(60px); }
.bp3-tooltip{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  -webkit-transform:scale(1);
          transform:scale(1); }
  .bp3-tooltip .bp3-popover-arrow{
    height:22px;
    position:absolute;
    width:22px; }
    .bp3-tooltip .bp3-popover-arrow::before{
      height:14px;
      margin:4px;
      width:14px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip{
    margin-bottom:11px;
    margin-top:-11px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow{
      bottom:-8px; }
      .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(-90deg);
                transform:rotate(-90deg); }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip{
    margin-left:11px; }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow{
      left:-8px; }
      .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(0);
                transform:rotate(0); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip{
    margin-top:11px; }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow{
      top:-8px; }
      .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(90deg);
                transform:rotate(90deg); }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip{
    margin-left:-11px;
    margin-right:11px; }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow{
      right:-8px; }
      .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(180deg);
                transform:rotate(180deg); }
  .bp3-tether-element-attached-middle > .bp3-tooltip > .bp3-popover-arrow{
    top:50%;
    -webkit-transform:translateY(-50%);
            transform:translateY(-50%); }
  .bp3-tether-element-attached-center > .bp3-tooltip > .bp3-popover-arrow{
    right:50%;
    -webkit-transform:translateX(50%);
            transform:translateX(50%); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow{
    top:-0.22183px; }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow{
    right:-0.22183px; }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow{
    left:-0.22183px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow{
    bottom:-0.22183px; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:top left;
            transform-origin:top left; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:top center;
            transform-origin:top center; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:top right;
            transform-origin:top right; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:center left;
            transform-origin:center left; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:center center;
            transform-origin:center center; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:center right;
            transform-origin:center right; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:bottom left;
            transform-origin:bottom left; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:bottom center;
            transform-origin:bottom center; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:bottom right;
            transform-origin:bottom right; }
  .bp3-tooltip .bp3-popover-content{
    background:#394b59;
    color:#f5f8fa; }
  .bp3-tooltip .bp3-popover-arrow::before{
    -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2);
            box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2); }
  .bp3-tooltip .bp3-popover-arrow-border{
    fill:#10161a;
    fill-opacity:0.1; }
  .bp3-tooltip .bp3-popover-arrow-fill{
    fill:#394b59; }
  .bp3-popover-enter > .bp3-tooltip, .bp3-popover-appear > .bp3-tooltip{
    -webkit-transform:scale(0.8);
            transform:scale(0.8); }
  .bp3-popover-enter-active > .bp3-tooltip, .bp3-popover-appear-active > .bp3-tooltip{
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-popover-exit > .bp3-tooltip{
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-popover-exit-active > .bp3-tooltip{
    -webkit-transform:scale(0.8);
            transform:scale(0.8);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-tooltip .bp3-popover-content{
    padding:10px 12px; }
  .bp3-tooltip.bp3-dark,
  .bp3-dark .bp3-tooltip{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-tooltip.bp3-dark .bp3-popover-content,
    .bp3-dark .bp3-tooltip .bp3-popover-content{
      background:#e1e8ed;
      color:#394b59; }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow::before,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow::before{
      -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4);
              box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4); }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow-border,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow-border{
      fill:#10161a;
      fill-opacity:0.2; }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow-fill,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow-fill{
      fill:#e1e8ed; }
  .bp3-tooltip.bp3-intent-primary .bp3-popover-content{
    background:#137cbd;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-primary .bp3-popover-arrow-fill{
    fill:#137cbd; }
  .bp3-tooltip.bp3-intent-success .bp3-popover-content{
    background:#0f9960;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-success .bp3-popover-arrow-fill{
    fill:#0f9960; }
  .bp3-tooltip.bp3-intent-warning .bp3-popover-content{
    background:#d9822b;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-warning .bp3-popover-arrow-fill{
    fill:#d9822b; }
  .bp3-tooltip.bp3-intent-danger .bp3-popover-content{
    background:#db3737;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-danger .bp3-popover-arrow-fill{
    fill:#db3737; }

.bp3-tooltip-indicator{
  border-bottom:dotted 1px;
  cursor:help; }
.bp3-tree .bp3-icon, .bp3-tree .bp3-icon-standard, .bp3-tree .bp3-icon-large{
  color:#5c7080; }
  .bp3-tree .bp3-icon.bp3-intent-primary, .bp3-tree .bp3-icon-standard.bp3-intent-primary, .bp3-tree .bp3-icon-large.bp3-intent-primary{
    color:#137cbd; }
  .bp3-tree .bp3-icon.bp3-intent-success, .bp3-tree .bp3-icon-standard.bp3-intent-success, .bp3-tree .bp3-icon-large.bp3-intent-success{
    color:#0f9960; }
  .bp3-tree .bp3-icon.bp3-intent-warning, .bp3-tree .bp3-icon-standard.bp3-intent-warning, .bp3-tree .bp3-icon-large.bp3-intent-warning{
    color:#d9822b; }
  .bp3-tree .bp3-icon.bp3-intent-danger, .bp3-tree .bp3-icon-standard.bp3-intent-danger, .bp3-tree .bp3-icon-large.bp3-intent-danger{
    color:#db3737; }

.bp3-tree-node-list{
  list-style:none;
  margin:0;
  padding-left:0; }

.bp3-tree-root{
  background-color:transparent;
  cursor:default;
  padding-left:0;
  position:relative; }

.bp3-tree-node-content-0{
  padding-left:0px; }

.bp3-tree-node-content-1{
  padding-left:23px; }

.bp3-tree-node-content-2{
  padding-left:46px; }

.bp3-tree-node-content-3{
  padding-left:69px; }

.bp3-tree-node-content-4{
  padding-left:92px; }

.bp3-tree-node-content-5{
  padding-left:115px; }

.bp3-tree-node-content-6{
  padding-left:138px; }

.bp3-tree-node-content-7{
  padding-left:161px; }

.bp3-tree-node-content-8{
  padding-left:184px; }

.bp3-tree-node-content-9{
  padding-left:207px; }

.bp3-tree-node-content-10{
  padding-left:230px; }

.bp3-tree-node-content-11{
  padding-left:253px; }

.bp3-tree-node-content-12{
  padding-left:276px; }

.bp3-tree-node-content-13{
  padding-left:299px; }

.bp3-tree-node-content-14{
  padding-left:322px; }

.bp3-tree-node-content-15{
  padding-left:345px; }

.bp3-tree-node-content-16{
  padding-left:368px; }

.bp3-tree-node-content-17{
  padding-left:391px; }

.bp3-tree-node-content-18{
  padding-left:414px; }

.bp3-tree-node-content-19{
  padding-left:437px; }

.bp3-tree-node-content-20{
  padding-left:460px; }

.bp3-tree-node-content{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  height:30px;
  padding-right:5px;
  width:100%; }
  .bp3-tree-node-content:hover{
    background-color:rgba(191, 204, 214, 0.4); }

.bp3-tree-node-caret,
.bp3-tree-node-caret-none{
  min-width:30px; }

.bp3-tree-node-caret{
  color:#5c7080;
  cursor:pointer;
  padding:7px;
  -webkit-transform:rotate(0deg);
          transform:rotate(0deg);
  -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-tree-node-caret:hover{
    color:#182026; }
  .bp3-dark .bp3-tree-node-caret{
    color:#a7b6c2; }
    .bp3-dark .bp3-tree-node-caret:hover{
      color:#f5f8fa; }
  .bp3-tree-node-caret.bp3-tree-node-caret-open{
    -webkit-transform:rotate(90deg);
            transform:rotate(90deg); }
  .bp3-tree-node-caret.bp3-icon-standard::before{
    content:""; }

.bp3-tree-node-icon{
  margin-right:7px;
  position:relative; }

.bp3-tree-node-label{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  position:relative;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-tree-node-label span{
    display:inline; }

.bp3-tree-node-secondary-label{
  padding:0 5px;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-tree-node-secondary-label .bp3-popover-wrapper,
  .bp3-tree-node-secondary-label .bp3-popover-target{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex; }

.bp3-tree-node.bp3-disabled .bp3-tree-node-content{
  background-color:inherit;
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-tree-node.bp3-disabled .bp3-tree-node-caret,
.bp3-tree-node.bp3-disabled .bp3-tree-node-icon{
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content{
  background-color:#137cbd; }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content,
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon, .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon-standard, .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon-large{
    color:#ffffff; }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-tree-node-caret::before{
    color:rgba(255, 255, 255, 0.7); }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-tree-node-caret:hover::before{
    color:#ffffff; }

.bp3-dark .bp3-tree-node-content:hover{
  background-color:rgba(92, 112, 128, 0.3); }

.bp3-dark .bp3-tree .bp3-icon, .bp3-dark .bp3-tree .bp3-icon-standard, .bp3-dark .bp3-tree .bp3-icon-large{
  color:#a7b6c2; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-primary, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-primary, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-primary{
    color:#137cbd; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-success, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-success, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-success{
    color:#0f9960; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-warning, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-warning, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-warning{
    color:#d9822b; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-danger, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-danger, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-danger{
    color:#db3737; }

.bp3-dark .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content{
  background-color:#137cbd; }
.bp3-omnibar{
  -webkit-filter:blur(0);
          filter:blur(0);
  opacity:1;
  background-color:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  left:calc(50% - 250px);
  top:20vh;
  width:500px;
  z-index:21; }
  .bp3-omnibar.bp3-overlay-enter, .bp3-omnibar.bp3-overlay-appear{
    -webkit-filter:blur(20px);
            filter:blur(20px);
    opacity:0.2; }
  .bp3-omnibar.bp3-overlay-enter-active, .bp3-omnibar.bp3-overlay-appear-active{
    -webkit-filter:blur(0);
            filter:blur(0);
    opacity:1;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:filter, opacity;
    transition-property:filter, opacity, -webkit-filter;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-omnibar.bp3-overlay-exit{
    -webkit-filter:blur(0);
            filter:blur(0);
    opacity:1; }
  .bp3-omnibar.bp3-overlay-exit-active{
    -webkit-filter:blur(20px);
            filter:blur(20px);
    opacity:0.2;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:filter, opacity;
    transition-property:filter, opacity, -webkit-filter;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-omnibar .bp3-input{
    background-color:transparent;
    border-radius:0; }
    .bp3-omnibar .bp3-input, .bp3-omnibar .bp3-input:focus{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-omnibar .bp3-menu{
    background-color:transparent;
    border-radius:0;
    -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
    max-height:calc(60vh - 40px);
    overflow:auto; }
    .bp3-omnibar .bp3-menu:empty{
      display:none; }
  .bp3-dark .bp3-omnibar, .bp3-omnibar.bp3-dark{
    background-color:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4); }

.bp3-omnibar-overlay .bp3-overlay-backdrop{
  background-color:rgba(16, 22, 26, 0.2); }

.bp3-select-popover .bp3-popover-content{
  padding:5px; }

.bp3-select-popover .bp3-input-group{
  margin-bottom:0; }

.bp3-select-popover .bp3-menu{
  max-height:300px;
  max-width:400px;
  overflow:auto;
  padding:0; }
  .bp3-select-popover .bp3-menu:not(:first-child){
    padding-top:5px; }

.bp3-multi-select{
  min-width:150px; }

.bp3-multi-select-popover .bp3-menu{
  max-height:300px;
  max-width:400px;
  overflow:auto; }

.bp3-select-popover .bp3-popover-content{
  padding:5px; }

.bp3-select-popover .bp3-input-group{
  margin-bottom:0; }

.bp3-select-popover .bp3-menu{
  max-height:300px;
  max-width:400px;
  overflow:auto;
  padding:0; }
  .bp3-select-popover .bp3-menu:not(:first-child){
    padding-top:5px; }
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensureUiComponents() in @jupyterlab/buildutils */

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

/* Icons urls */

:root {
  --jp-icon-add: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDEzaC02djZoLTJ2LTZINXYtMmg2VjVoMnY2aDZ2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bug: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDhoLTIuODFjLS40NS0uNzgtMS4wNy0xLjQ1LTEuODItMS45NkwxNyA0LjQxIDE1LjU5IDNsLTIuMTcgMi4xN0MxMi45NiA1LjA2IDEyLjQ5IDUgMTIgNWMtLjQ5IDAtLjk2LjA2LTEuNDEuMTdMOC40MSAzIDcgNC40MWwxLjYyIDEuNjNDNy44OCA2LjU1IDcuMjYgNy4yMiA2LjgxIDhINHYyaDIuMDljLS4wNS4zMy0uMDkuNjYtLjA5IDF2MUg0djJoMnYxYzAgLjM0LjA0LjY3LjA5IDFINHYyaDIuODFjMS4wNCAxLjc5IDIuOTcgMyA1LjE5IDNzNC4xNS0xLjIxIDUuMTktM0gyMHYtMmgtMi4wOWMuMDUtLjMzLjA5LS42Ni4wOS0xdi0xaDJ2LTJoLTJ2LTFjMC0uMzQtLjA0LS42Ny0uMDktMUgyMFY4em0tNiA4aC00di0yaDR2MnptMC00aC00di0yaDR2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-build: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE0LjkgMTcuNDVDMTYuMjUgMTcuNDUgMTcuMzUgMTYuMzUgMTcuMzUgMTVDMTcuMzUgMTMuNjUgMTYuMjUgMTIuNTUgMTQuOSAxMi41NUMxMy41NCAxMi41NSAxMi40NSAxMy42NSAxMi40NSAxNUMxMi40NSAxNi4zNSAxMy41NCAxNy40NSAxNC45IDE3LjQ1Wk0yMC4xIDE1LjY4TDIxLjU4IDE2Ljg0QzIxLjcxIDE2Ljk1IDIxLjc1IDE3LjEzIDIxLjY2IDE3LjI5TDIwLjI2IDE5LjcxQzIwLjE3IDE5Ljg2IDIwIDE5LjkyIDE5LjgzIDE5Ljg2TDE4LjA5IDE5LjE2QzE3LjczIDE5LjQ0IDE3LjMzIDE5LjY3IDE2LjkxIDE5Ljg1TDE2LjY0IDIxLjdDMTYuNjIgMjEuODcgMTYuNDcgMjIgMTYuMyAyMkgxMy41QzEzLjMyIDIyIDEzLjE4IDIxLjg3IDEzLjE1IDIxLjdMMTIuODkgMTkuODVDMTIuNDYgMTkuNjcgMTIuMDcgMTkuNDQgMTEuNzEgMTkuMTZMOS45NjAwMiAxOS44NkM5LjgxMDAyIDE5LjkyIDkuNjIwMDIgMTkuODYgOS41NDAwMiAxOS43MUw4LjE0MDAyIDE3LjI5QzguMDUwMDIgMTcuMTMgOC4wOTAwMiAxNi45NSA4LjIyMDAyIDE2Ljg0TDkuNzAwMDIgMTUuNjhMOS42NTAwMSAxNUw5LjcwMDAyIDE0LjMxTDguMjIwMDIgMTMuMTZDOC4wOTAwMiAxMy4wNSA4LjA1MDAyIDEyLjg2IDguMTQwMDIgMTIuNzFMOS41NDAwMiAxMC4yOUM5LjYyMDAyIDEwLjEzIDkuODEwMDIgMTAuMDcgOS45NjAwMiAxMC4xM0wxMS43MSAxMC44NEMxMi4wNyAxMC41NiAxMi40NiAxMC4zMiAxMi44OSAxMC4xNUwxMy4xNSA4LjI4OTk4QzEzLjE4IDguMTI5OTggMTMuMzIgNy45OTk5OCAxMy41IDcuOTk5OThIMTYuM0MxNi40NyA3Ljk5OTk4IDE2LjYyIDguMTI5OTggMTYuNjQgOC4yODk5OEwxNi45MSAxMC4xNUMxNy4zMyAxMC4zMiAxNy43MyAxMC41NiAxOC4wOSAxMC44NEwxOS44MyAxMC4xM0MyMCAxMC4wNyAyMC4xNyAxMC4xMyAyMC4yNiAxMC4yOUwyMS42NiAxMi43MUMyMS43NSAxMi44NiAyMS43MSAxMy4wNSAyMS41OCAxMy4xNkwyMC4xIDE0LjMxTDIwLjE1IDE1TDIwLjEgMTUuNjhaIi8+CiAgICA8cGF0aCBkPSJNNy4zMjk2NiA3LjQ0NDU0QzguMDgzMSA3LjAwOTU0IDguMzM5MzIgNi4wNTMzMiA3LjkwNDMyIDUuMjk5ODhDNy40NjkzMiA0LjU0NjQzIDYuNTA4MSA0LjI4MTU2IDUuNzU0NjYgNC43MTY1NkM1LjM5MTc2IDQuOTI2MDggNS4xMjY5NSA1LjI3MTE4IDUuMDE4NDkgNS42NzU5NEM0LjkxMDA0IDYuMDgwNzEgNC45NjY4MiA2LjUxMTk4IDUuMTc2MzQgNi44NzQ4OEM1LjYxMTM0IDcuNjI4MzIgNi41NzYyMiA3Ljg3OTU0IDcuMzI5NjYgNy40NDQ1NFpNOS42NTcxOCA0Ljc5NTkzTDEwLjg2NzIgNC45NTE3OUMxMC45NjI4IDQuOTc3NDEgMTEuMDQwMiA1LjA3MTMzIDExLjAzODIgNS4xODc5M0wxMS4wMzg4IDYuOTg4OTNDMTEuMDQ1NSA3LjEwMDU0IDEwLjk2MTYgNy4xOTUxOCAxMC44NTUgNy4yMTA1NEw5LjY2MDAxIDcuMzgwODNMOS4yMzkxNSA4LjEzMTg4TDkuNjY5NjEgOS4yNTc0NUM5LjcwNzI5IDkuMzYyNzEgOS42NjkzNCA5LjQ3Njk5IDkuNTc0MDggOS41MzE5OUw4LjAxNTIzIDEwLjQzMkM3LjkxMTMxIDEwLjQ5MiA3Ljc5MzM3IDEwLjQ2NzcgNy43MjEwNSAxMC4zODI0TDYuOTg3NDggOS40MzE4OEw2LjEwOTMxIDkuNDMwODNMNS4zNDcwNCAxMC4zOTA1QzUuMjg5MDkgMTAuNDcwMiA1LjE3MzgzIDEwLjQ5MDUgNS4wNzE4NyAxMC40MzM5TDMuNTEyNDUgOS41MzI5M0MzLjQxMDQ5IDkuNDc2MzMgMy4zNzY0NyA5LjM1NzQxIDMuNDEwNzUgOS4yNTY3OUwzLjg2MzQ3IDguMTQwOTNMMy42MTc0OSA3Ljc3NDg4TDMuNDIzNDcgNy4zNzg4M0wyLjIzMDc1IDcuMjEyOTdDMi4xMjY0NyA3LjE5MjM1IDIuMDQwNDkgNy4xMDM0MiAyLjA0MjQ1IDYuOTg2ODJMMi4wNDE4NyA1LjE4NTgyQzIuMDQzODMgNS4wNjkyMiAyLjExOTA5IDQuOTc5NTggMi4yMTcwNCA0Ljk2OTIyTDMuNDIwNjUgNC43OTM5M0wzLjg2NzQ5IDQuMDI3ODhMMy40MTEwNSAyLjkxNzMxQzMuMzczMzcgMi44MTIwNCAzLjQxMTMxIDIuNjk3NzYgMy41MTUyMyAyLjYzNzc2TDUuMDc0MDggMS43Mzc3NkM1LjE2OTM0IDEuNjgyNzYgNS4yODcyOSAxLjcwNzA0IDUuMzU5NjEgMS43OTIzMUw2LjExOTE1IDIuNzI3ODhMNi45ODAwMSAyLjczODkzTDcuNzI0OTYgMS43ODkyMkM3Ljc5MTU2IDEuNzA0NTggNy45MTU0OCAxLjY3OTIyIDguMDA4NzkgMS43NDA4Mkw5LjU2ODIxIDIuNjQxODJDOS42NzAxNyAyLjY5ODQyIDkuNzEyODUgMi44MTIzNCA5LjY4NzIzIDIuOTA3OTdMOS4yMTcxOCA0LjAzMzgzTDkuNDYzMTYgNC4zOTk4OEw5LjY1NzE4IDQuNzk1OTNaIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iOS45LDEzLjYgMy42LDcuNCA0LjQsNi42IDkuOSwxMi4yIDE1LjQsNi43IDE2LjEsNy40ICIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNS45TDksOS43bDMuOC0zLjhsMS4yLDEuMmwtNC45LDVsLTQuOS01TDUuMiw1Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNy41TDksMTEuMmwzLjgtMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-left: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik0xMC44LDEyLjhMNy4xLDlsMy44LTMuOGwwLDcuNkgxMC44eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-right: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik03LjIsNS4yTDEwLjksOWwtMy44LDMuOFY1LjJINy4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-up-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iMTUuNCwxMy4zIDkuOSw3LjcgNC40LDEzLjIgMy42LDEyLjUgOS45LDYuMyAxNi4xLDEyLjYgIi8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-up: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik01LjIsMTAuNUw5LDYuOGwzLjgsMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-case-sensitive: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWFjY2VudDIiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTcuNiw4aDAuOWwzLjUsOGgtMS4xTDEwLDE0SDZsLTAuOSwySDRMNy42LDh6IE04LDkuMUw2LjQsMTNoMy4yTDgsOS4xeiIvPgogICAgPHBhdGggZD0iTTE2LjYsOS44Yy0wLjIsMC4xLTAuNCwwLjEtMC43LDAuMWMtMC4yLDAtMC40LTAuMS0wLjYtMC4yYy0wLjEtMC4xLTAuMi0wLjQtMC4yLTAuNyBjLTAuMywwLjMtMC42LDAuNS0wLjksMC43Yy0wLjMsMC4xLTAuNywwLjItMS4xLDAuMmMtMC4zLDAtMC41LDAtMC43LTAuMWMtMC4yLTAuMS0wLjQtMC4yLTAuNi0wLjNjLTAuMi0wLjEtMC4zLTAuMy0wLjQtMC41IGMtMC4xLTAuMi0wLjEtMC40LTAuMS0wLjdjMC0wLjMsMC4xLTAuNiwwLjItMC44YzAuMS0wLjIsMC4zLTAuNCwwLjQtMC41QzEyLDcsMTIuMiw2LjksMTIuNSw2LjhjMC4yLTAuMSwwLjUtMC4xLDAuNy0wLjIgYzAuMy0wLjEsMC41LTAuMSwwLjctMC4xYzAuMiwwLDAuNC0wLjEsMC42LTAuMWMwLjIsMCwwLjMtMC4xLDAuNC0wLjJjMC4xLTAuMSwwLjItMC4yLDAuMi0wLjRjMC0xLTEuMS0xLTEuMy0xIGMtMC40LDAtMS40LDAtMS40LDEuMmgtMC45YzAtMC40LDAuMS0wLjcsMC4yLTFjMC4xLTAuMiwwLjMtMC40LDAuNS0wLjZjMC4yLTAuMiwwLjUtMC4zLDAuOC0wLjNDMTMuMyw0LDEzLjYsNCwxMy45LDQgYzAuMywwLDAuNSwwLDAuOCwwLjFjMC4zLDAsMC41LDAuMSwwLjcsMC4yYzAuMiwwLjEsMC40LDAuMywwLjUsMC41QzE2LDUsMTYsNS4yLDE2LDUuNnYyLjljMCwwLjIsMCwwLjQsMCwwLjUgYzAsMC4xLDAuMSwwLjIsMC4zLDAuMmMwLjEsMCwwLjIsMCwwLjMsMFY5Ljh6IE0xNS4yLDYuOWMtMS4yLDAuNi0zLjEsMC4yLTMuMSwxLjRjMCwxLjQsMy4xLDEsMy4xLTAuNVY2Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTYuMTdMNC44MyAxMmwtMS40MiAxLjQxTDkgMTkgMjEgN2wtMS40MS0xLjQxeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDJDNi40NyAyIDIgNi40NyAyIDEyczQuNDcgMTAgMTAgMTAgMTAtNC40NyAxMC0xMFMxNy41MyAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iOSIgY3k9IjkiIHI9IjgiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-clear: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8bWFzayBpZD0iZG9udXRIb2xlIj4KICAgIDxyZWN0IHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0id2hpdGUiIC8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI4IiBmaWxsPSJibGFjayIvPgogIDwvbWFzaz4KCiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxyZWN0IGhlaWdodD0iMTgiIHdpZHRoPSIyIiB4PSIxMSIgeT0iMyIgdHJhbnNmb3JtPSJyb3RhdGUoMzE1LCAxMiwgMTIpIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgbWFzaz0idXJsKCNkb251dEhvbGUpIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-close: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1ub25lIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIGpwLWljb24zLWhvdmVyIiBmaWxsPSJub25lIj4KICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjExIi8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIGpwLWljb24tYWNjZW50Mi1ob3ZlciIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMTkgNi40MUwxNy41OSA1IDEyIDEwLjU5IDYuNDEgNSA1IDYuNDEgMTAuNTkgMTIgNSAxNy41OSA2LjQxIDE5IDEyIDEzLjQxIDE3LjU5IDE5IDE5IDE3LjU5IDEzLjQxIDEyeiIvPgogIDwvZz4KCiAgPGcgY2xhc3M9ImpwLWljb24tbm9uZSBqcC1pY29uLWJ1c3kiIGZpbGw9Im5vbmUiPgogICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTExLjQgMTguNkw2LjggMTRMMTEuNCA5LjRMMTAgOEw0IDE0TDEwIDIwTDExLjQgMTguNlpNMTYuNiAxOC42TDIxLjIgMTRMMTYuNiA5LjRMMTggOEwyNCAxNEwxOCAyMEwxNi42IDE4LjZWMTguNloiLz4KCTwvZz4KPC9zdmc+Cg==);
  --jp-icon-console: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwMCAyMDAiPgogIDxnIGNsYXNzPSJqcC1pY29uLWJyYW5kMSBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMjg4RDEiPgogICAgPHBhdGggZD0iTTIwIDE5LjhoMTYwdjE1OS45SDIweiIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNmZmYiPgogICAgPHBhdGggZD0iTTEwNSAxMjcuM2g0MHYxMi44aC00MHpNNTEuMSA3N0w3NCA5OS45bC0yMy4zIDIzLjMgMTAuNSAxMC41IDIzLjMtMjMuM0w5NSA5OS45IDg0LjUgODkuNCA2MS42IDY2LjV6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-copy: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTExLjksMUgzLjJDMi40LDEsMS43LDEuNywxLjcsMi41djEwLjJoMS41VjIuNWg4LjdWMXogTTE0LjEsMy45aC04Yy0wLjgsMC0xLjUsMC43LTEuNSwxLjV2MTAuMmMwLDAuOCwwLjcsMS41LDEuNSwxLjVoOCBjMC44LDAsMS41LTAuNywxLjUtMS41VjUuNEMxNS41LDQuNiwxNC45LDMuOSwxNC4xLDMuOXogTTE0LjEsMTUuNWgtOFY1LjRoOFYxNS41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-cut: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkuNjQgNy42NGMuMjMtLjUuMzYtMS4wNS4zNi0xLjY0IDAtMi4yMS0xLjc5LTQtNC00UzIgMy43OSAyIDZzMS43OSA0IDQgNGMuNTkgMCAxLjE0LS4xMyAxLjY0LS4zNkwxMCAxMmwtMi4zNiAyLjM2QzcuMTQgMTQuMTMgNi41OSAxNCA2IDE0Yy0yLjIxIDAtNCAxLjc5LTQgNHMxLjc5IDQgNCA0IDQtMS43OSA0LTRjMC0uNTktLjEzLTEuMTQtLjM2LTEuNjRMMTIgMTRsNyA3aDN2LTFMOS42NCA3LjY0ek02IDhjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTAgMTJjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTYtNy41Yy0uMjggMC0uNS0uMjItLjUtLjVzLjIyLS41LjUtLjUuNS4yMi41LjUtLjIyLjUtLjUuNXpNMTkgM2wtNiA2IDIgMiA3LTdWM3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-download: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDloLTRWM0g5djZINWw3IDcgNy03ek01IDE4djJoMTR2LTJINXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-edit: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMgMTcuMjVWMjFoMy43NUwxNy44MSA5Ljk0bC0zLjc1LTMuNzVMMyAxNy4yNXpNMjAuNzEgNy4wNGMuMzktLjM5LjM5LTEuMDIgMC0xLjQxbC0yLjM0LTIuMzRjLS4zOS0uMzktMS4wMi0uMzktMS40MSAwbC0xLjgzIDEuODMgMy43NSAzLjc1IDEuODMtMS44M3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-ellipses: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iNSIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxOSIgY3k9IjEyIiByPSIyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-extension: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwLjUgMTFIMTlWN2MwLTEuMS0uOS0yLTItMmgtNFYzLjVDMTMgMi4xMiAxMS44OCAxIDEwLjUgMVM4IDIuMTIgOCAzLjVWNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAydjMuOEgzLjVjMS40OSAwIDIuNyAxLjIxIDIuNyAyLjdzLTEuMjEgMi43LTIuNyAyLjdIMlYyMGMwIDEuMS45IDIgMiAyaDMuOHYtMS41YzAtMS40OSAxLjIxLTIuNyAyLjctMi43IDEuNDkgMCAyLjcgMS4yMSAyLjcgMi43VjIySDE3YzEuMSAwIDItLjkgMi0ydi00aDEuNWMxLjM4IDAgMi41LTEuMTIgMi41LTIuNVMyMS44OCAxMSAyMC41IDExeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-fast-forward: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTQgMThsOC41LTZMNCA2djEyem05LTEydjEybDguNS02TDEzIDZ6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-file-upload: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTZoNnYtNmg0bC03LTctNyA3aDR6bS00IDJoMTR2Mkg1eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-file: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuMyA4LjJsLTUuNS01LjVjLS4zLS4zLS43LS41LTEuMi0uNUgzLjljLS44LjEtMS42LjktMS42IDEuOHYxNC4xYzAgLjkuNyAxLjYgMS42IDEuNmgxNC4yYy45IDAgMS42LS43IDEuNi0xLjZWOS40Yy4xLS41LS4xLS45LS40LTEuMnptLTUuOC0zLjNsMy40IDMuNmgtMy40VjQuOXptMy45IDEyLjdINC43Yy0uMSAwLS4yIDAtLjItLjJWNC43YzAtLjIuMS0uMy4yLS4zaDcuMnY0LjRzMCAuOC4zIDEuMWMuMy4zIDEuMS4zIDEuMS4zaDQuM3Y3LjJzLS4xLjItLjIuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-filter-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEwIDE4aDR2LTJoLTR2MnpNMyA2djJoMThWNkgzem0zIDdoMTJ2LTJINnYyeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY4YzAtMS4xLS45LTItMi0yaC04bC0yLTJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-html5: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMDAiIGQ9Ik0xMDguNCAwaDIzdjIyLjhoMjEuMlYwaDIzdjY5aC0yM1Y0NmgtMjF2MjNoLTIzLjJNMjA2IDIzaC0yMC4zVjBoNjMuN3YyM0gyMjl2NDZoLTIzbTUzLjUtNjloMjQuMWwxNC44IDI0LjNMMzEzLjIgMGgyNC4xdjY5aC0yM1YzNC44bC0xNi4xIDI0LjgtMTYuMS0yNC44VjY5aC0yMi42bTg5LjItNjloMjN2NDYuMmgzMi42VjY5aC01NS42Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2U0NGQyNiIgZD0iTTEwNy42IDQ3MWwtMzMtMzcwLjRoMzYyLjhsLTMzIDM3MC4yTDI1NS43IDUxMiIvPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNmMTY1MjkiIGQ9Ik0yNTYgNDgwLjVWMTMxaDE0OC4zTDM3NiA0NDciLz4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNlYmViZWIiIGQ9Ik0xNDIgMTc2LjNoMTE0djQ1LjRoLTY0LjJsNC4yIDQ2LjVoNjB2NDUuM0gxNTQuNG0yIDIyLjhIMjAybDMuMiAzNi4zIDUwLjggMTMuNnY0Ny40bC05My4yLTI2Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIiBkPSJNMzY5LjYgMTc2LjNIMjU1Ljh2NDUuNGgxMDkuNm0tNC4xIDQ2LjVIMjU1Ljh2NDUuNGg1NmwtNS4zIDU5LTUwLjcgMTMuNnY0Ny4ybDkzLTI1LjgiLz4KPC9zdmc+Cg==);
  --jp-icon-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1icmFuZDQganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNGRkYiIGQ9Ik0yLjIgMi4yaDE3LjV2MTcuNUgyLjJ6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzNGNTFCNSIgZD0iTTIuMiAyLjJ2MTcuNWgxNy41bC4xLTE3LjVIMi4yem0xMi4xIDIuMmMxLjIgMCAyLjIgMSAyLjIgMi4ycy0xIDIuMi0yLjIgMi4yLTIuMi0xLTIuMi0yLjIgMS0yLjIgMi4yLTIuMnpNNC40IDE3LjZsMy4zLTguOCAzLjMgNi42IDIuMi0zLjIgNC40IDUuNEg0LjR6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-inspector: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0tNSAxNEg0di00aDExdjR6bTAtNUg0VjloMTF2NHptNSA1aC00VjloNHY5eiIvPgo8L3N2Zz4K);
  --jp-icon-json: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMSBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNGOUE4MjUiPgogICAgPHBhdGggZD0iTTIwLjIgMTEuOGMtMS42IDAtMS43LjUtMS43IDEgMCAuNC4xLjkuMSAxLjMuMS41LjEuOS4xIDEuMyAwIDEuNy0xLjQgMi4zLTMuNSAyLjNoLS45di0xLjloLjVjMS4xIDAgMS40IDAgMS40LS44IDAtLjMgMC0uNi0uMS0xIDAtLjQtLjEtLjgtLjEtMS4yIDAtMS4zIDAtMS44IDEuMy0yLTEuMy0uMi0xLjMtLjctMS4zLTIgMC0uNC4xLS44LjEtMS4yLjEtLjQuMS0uNy4xLTEgMC0uOC0uNC0uNy0xLjQtLjhoLS41VjQuMWguOWMyLjIgMCAzLjUuNyAzLjUgMi4zIDAgLjQtLjEuOS0uMSAxLjMtLjEuNS0uMS45LS4xIDEuMyAwIC41LjIgMSAxLjcgMXYxLjh6TTEuOCAxMC4xYzEuNiAwIDEuNy0uNSAxLjctMSAwLS40LS4xLS45LS4xLTEuMy0uMS0uNS0uMS0uOS0uMS0xLjMgMC0xLjYgMS40LTIuMyAzLjUtMi4zaC45djEuOWgtLjVjLTEgMC0xLjQgMC0xLjQuOCAwIC4zIDAgLjYuMSAxIDAgLjIuMS42LjEgMSAwIDEuMyAwIDEuOC0xLjMgMkM2IDExLjIgNiAxMS43IDYgMTNjMCAuNC0uMS44LS4xIDEuMi0uMS4zLS4xLjctLjEgMSAwIC44LjMuOCAxLjQuOGguNXYxLjloLS45Yy0yLjEgMC0zLjUtLjYtMy41LTIuMyAwLS40LjEtLjkuMS0xLjMuMS0uNS4xLS45LjEtMS4zIDAtLjUtLjItMS0xLjctMXYtMS45eiIvPgogICAgPGNpcmNsZSBjeD0iMTEiIGN5PSIxMy44IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY3g9IjExIiBjeT0iOC4yIiByPSIyLjEiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter-favicon: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUyIiBoZWlnaHQ9IjE2NSIgdmlld0JveD0iMCAwIDE1MiAxNjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA3ODk0NywgMTEwLjU4MjkyNykiIGQ9Ik03NS45NDIyODQyLDI5LjU4MDQ1NjEgQzQzLjMwMjM5NDcsMjkuNTgwNDU2MSAxNC43OTY3ODMyLDE3LjY1MzQ2MzQgMCwwIEM1LjUxMDgzMjExLDE1Ljg0MDY4MjkgMTUuNzgxNTM4OSwyOS41NjY3NzMyIDI5LjM5MDQ5NDcsMzkuMjc4NDE3MSBDNDIuOTk5Nyw0OC45ODk4NTM3IDU5LjI3MzcsNTQuMjA2NzgwNSA3NS45NjA1Nzg5LDU0LjIwNjc4MDUgQzkyLjY0NzQ1NzksNTQuMjA2NzgwNSAxMDguOTIxNDU4LDQ4Ljk4OTg1MzcgMTIyLjUzMDY2MywzOS4yNzg0MTcxIEMxMzYuMTM5NDUzLDI5LjU2Njc3MzIgMTQ2LjQxMDI4NCwxNS44NDA2ODI5IDE1MS45MjExNTgsMCBDMTM3LjA4Nzg2OCwxNy42NTM0NjM0IDEwOC41ODI1ODksMjkuNTgwNDU2MSA3NS45NDIyODQyLDI5LjU4MDQ1NjEgTDc1Ljk0MjI4NDIsMjkuNTgwNDU2MSBaIiAvPgogICAgPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMzczNjgsIDAuNzA0ODc4KSIgZD0iTTc1Ljk3ODQ1NzksMjQuNjI2NDA3MyBDMTA4LjYxODc2MywyNC42MjY0MDczIDEzNy4xMjQ0NTgsMzYuNTUzNDQxNSAxNTEuOTIxMTU4LDU0LjIwNjc4MDUgQzE0Ni40MTAyODQsMzguMzY2MjIyIDEzNi4xMzk0NTMsMjQuNjQwMTMxNyAxMjIuNTMwNjYzLDE0LjkyODQ4NzggQzEwOC45MjE0NTgsNS4yMTY4NDM5IDkyLjY0NzQ1NzksMCA3NS45NjA1Nzg5LDAgQzU5LjI3MzcsMCA0Mi45OTk3LDUuMjE2ODQzOSAyOS4zOTA0OTQ3LDE0LjkyODQ4NzggQzE1Ljc4MTUzODksMjQuNjQwMTMxNyA1LjUxMDgzMjExLDM4LjM2NjIyMiAwLDU0LjIwNjc4MDUgQzE0LjgzMzA4MTYsMzYuNTg5OTI5MyA0My4zMzg1Njg0LDI0LjYyNjQwNzMgNzUuOTc4NDU3OSwyNC42MjY0MDczIEw3NS45Nzg0NTc5LDI0LjYyNjQwNzMgWiIgLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzkiIGhlaWdodD0iNTEiIHZpZXdCb3g9IjAgMCAzOSA1MSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTYzOCAtMjI4MSkiPgogICAgPGcgY2xhc3M9ImpwLWljb24td2FybjAiIGZpbGw9IiNGMzc3MjYiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5Ljc0IDIzMTEuOTgpIiBkPSJNIDE4LjI2NDYgNy4xMzQxMUMgMTAuNDE0NSA3LjEzNDExIDMuNTU4NzIgNC4yNTc2IDAgMEMgMS4zMjUzOSAzLjgyMDQgMy43OTU1NiA3LjEzMDgxIDcuMDY4NiA5LjQ3MzAzQyAxMC4zNDE3IDExLjgxNTIgMTQuMjU1NyAxMy4wNzM0IDE4LjI2OSAxMy4wNzM0QyAyMi4yODIzIDEzLjA3MzQgMjYuMTk2MyAxMS44MTUyIDI5LjQ2OTQgOS40NzMwM0MgMzIuNzQyNCA3LjEzMDgxIDM1LjIxMjYgMy44MjA0IDM2LjUzOCAwQyAzMi45NzA1IDQuMjU3NiAyNi4xMTQ4IDcuMTM0MTEgMTguMjY0NiA3LjEzNDExWiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5LjczIDIyODUuNDgpIiBkPSJNIDE4LjI3MzMgNS45MzkzMUMgMjYuMTIzNSA1LjkzOTMxIDMyLjk3OTMgOC44MTU4MyAzNi41MzggMTMuMDczNEMgMzUuMjEyNiA5LjI1MzAzIDMyLjc0MjQgNS45NDI2MiAyOS40Njk0IDMuNjAwNEMgMjYuMTk2MyAxLjI1ODE4IDIyLjI4MjMgMCAxOC4yNjkgMEMgMTQuMjU1NyAwIDEwLjM0MTcgMS4yNTgxOCA3LjA2ODYgMy42MDA0QyAzLjc5NTU2IDUuOTQyNjIgMS4zMjUzOSA5LjI1MzAzIDAgMTMuMDczNEMgMy41Njc0NSA4LjgyNDYzIDEwLjQyMzIgNS45MzkzMSAxOC4yNzMzIDUuOTM5MzFaIi8+CiAgICA8L2c+CiAgICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjY5LjMgMjI4MS4zMSkiIGQ9Ik0gNS44OTM1MyAyLjg0NEMgNS45MTg4OSAzLjQzMTY1IDUuNzcwODUgNC4wMTM2NyA1LjQ2ODE1IDQuNTE2NDVDIDUuMTY1NDUgNS4wMTkyMiA0LjcyMTY4IDUuNDIwMTUgNC4xOTI5OSA1LjY2ODUxQyAzLjY2NDMgNS45MTY4OCAzLjA3NDQ0IDYuMDAxNTEgMi40OTgwNSA1LjkxMTcxQyAxLjkyMTY2IDUuODIxOSAxLjM4NDYzIDUuNTYxNyAwLjk1NDg5OCA1LjE2NDAxQyAwLjUyNTE3IDQuNzY2MzMgMC4yMjIwNTYgNC4yNDkwMyAwLjA4MzkwMzcgMy42Nzc1N0MgLTAuMDU0MjQ4MyAzLjEwNjExIC0wLjAyMTIzIDIuNTA2MTcgMC4xNzg3ODEgMS45NTM2NEMgMC4zNzg3OTMgMS40MDExIDAuNzM2ODA5IDAuOTIwODE3IDEuMjA3NTQgMC41NzM1MzhDIDEuNjc4MjYgMC4yMjYyNTkgMi4yNDA1NSAwLjAyNzU5MTkgMi44MjMyNiAwLjAwMjY3MjI5QyAzLjYwMzg5IC0wLjAzMDcxMTUgNC4zNjU3MyAwLjI0OTc4OSA0Ljk0MTQyIDAuNzgyNTUxQyA1LjUxNzExIDEuMzE1MzEgNS44NTk1NiAyLjA1Njc2IDUuODkzNTMgMi44NDRaIi8+CiAgICAgIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2MzkuOCAyMzIzLjgxKSIgZD0iTSA3LjQyNzg5IDMuNTgzMzhDIDcuNDYwMDggNC4zMjQzIDcuMjczNTUgNS4wNTgxOSA2Ljg5MTkzIDUuNjkyMTNDIDYuNTEwMzEgNi4zMjYwNyA1Ljk1MDc1IDYuODMxNTYgNS4yODQxMSA3LjE0NDZDIDQuNjE3NDcgNy40NTc2MyAzLjg3MzcxIDcuNTY0MTQgMy4xNDcwMiA3LjQ1MDYzQyAyLjQyMDMyIDcuMzM3MTIgMS43NDMzNiA3LjAwODcgMS4yMDE4NCA2LjUwNjk1QyAwLjY2MDMyOCA2LjAwNTIgMC4yNzg2MSA1LjM1MjY4IDAuMTA1MDE3IDQuNjMyMDJDIC0wLjA2ODU3NTcgMy45MTEzNSAtMC4wMjYyMzYxIDMuMTU0OTQgMC4yMjY2NzUgMi40NTg1NkMgMC40Nzk1ODcgMS43NjIxNyAwLjkzMTY5NyAxLjE1NzEzIDEuNTI1NzYgMC43MjAwMzNDIDIuMTE5ODMgMC4yODI5MzUgMi44MjkxNCAwLjAzMzQzOTUgMy41NjM4OSAwLjAwMzEzMzQ0QyA0LjU0NjY3IC0wLjAzNzQwMzMgNS41MDUyOSAwLjMxNjcwNiA2LjIyOTYxIDAuOTg3ODM1QyA2Ljk1MzkzIDEuNjU4OTYgNy4zODQ4NCAyLjU5MjM1IDcuNDI3ODkgMy41ODMzOEwgNy40Mjc4OSAzLjU4MzM4WiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM4LjM2IDIyODYuMDYpIiBkPSJNIDIuMjc0NzEgNC4zOTYyOUMgMS44NDM2MyA0LjQxNTA4IDEuNDE2NzEgNC4zMDQ0NSAxLjA0Nzk5IDQuMDc4NDNDIDAuNjc5MjY4IDMuODUyNCAwLjM4NTMyOCAzLjUyMTE0IDAuMjAzMzcxIDMuMTI2NTZDIDAuMDIxNDEzNiAyLjczMTk4IC0wLjA0MDM3OTggMi4yOTE4MyAwLjAyNTgxMTYgMS44NjE4MUMgMC4wOTIwMDMxIDEuNDMxOCAwLjI4MzIwNCAxLjAzMTI2IDAuNTc1MjEzIDAuNzEwODgzQyAwLjg2NzIyMiAwLjM5MDUxIDEuMjQ2OTEgMC4xNjQ3MDggMS42NjYyMiAwLjA2MjA1OTJDIDIuMDg1NTMgLTAuMDQwNTg5NyAyLjUyNTYxIC0wLjAxNTQ3MTQgMi45MzA3NiAwLjEzNDIzNUMgMy4zMzU5MSAwLjI4Mzk0MSAzLjY4NzkyIDAuNTUxNTA1IDMuOTQyMjIgMC45MDMwNkMgNC4xOTY1MiAxLjI1NDYyIDQuMzQxNjkgMS42NzQzNiA0LjM1OTM1IDIuMTA5MTZDIDQuMzgyOTkgMi42OTEwNyA0LjE3Njc4IDMuMjU4NjkgMy43ODU5NyAzLjY4NzQ2QyAzLjM5NTE2IDQuMTE2MjQgMi44NTE2NiA0LjM3MTE2IDIuMjc0NzEgNC4zOTYyOUwgMi4yNzQ3MSA0LjM5NjI5WiIvPgogICAgPC9nPgogIDwvZz4+Cjwvc3ZnPgo=);
  --jp-icon-jupyterlab-wordmark: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIHZpZXdCb3g9IjAgMCAxODYwLjggNDc1Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0RTRFNEUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ4MC4xMzY0MDEsIDY0LjI3MTQ5MykiPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDU4Ljg3NTU2NikiPgogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA4NzYwMywgMC4xNDAyOTQpIj4KICAgICAgICA8cGF0aCBkPSJNLTQyNi45LDE2OS44YzAsNDguNy0zLjcsNjQuNy0xMy42LDc2LjRjLTEwLjgsMTAtMjUsMTUuNS0zOS43LDE1LjVsMy43LDI5IGMyMi44LDAuMyw0NC44LTcuOSw2MS45LTIzLjFjMTcuOC0xOC41LDI0LTQ0LjEsMjQtODMuM1YwSC00Mjd2MTcwLjFMLTQyNi45LDE2OS44TC00MjYuOSwxNjkuOHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTU1LjA0NTI5NiwgNTYuODM3MTA0KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTYyNDUzLCAxLjc5OTg0MikiPgogICAgICAgIDxwYXRoIGQ9Ik0tMzEyLDE0OGMwLDIxLDAsMzkuNSwxLjcsNTUuNGgtMzEuOGwtMi4xLTMzLjNoLTAuOGMtNi43LDExLjYtMTYuNCwyMS4zLTI4LDI3LjkgYy0xMS42LDYuNi0yNC44LDEwLTM4LjIsOS44Yy0zMS40LDAtNjktMTcuNy02OS04OVYwaDM2LjR2MTEyLjdjMCwzOC43LDExLjYsNjQuNyw0NC42LDY0LjdjMTAuMy0wLjIsMjAuNC0zLjUsMjguOS05LjQgYzguNS01LjksMTUuMS0xNC4zLDE4LjktMjMuOWMyLjItNi4xLDMuMy0xMi41LDMuMy0xOC45VjAuMmgzNi40VjE0OEgtMzEyTC0zMTIsMTQ4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzOTAuMDEzMzIyLCA1My40Nzk2MzgpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS43MDY0NTgsIDAuMjMxNDI1KSI+CiAgICAgICAgPHBhdGggZD0iTS00NzguNiw3MS40YzAtMjYtMC44LTQ3LTEuNy02Ni43aDMyLjdsMS43LDM0LjhoMC44YzcuMS0xMi41LDE3LjUtMjIuOCwzMC4xLTI5LjcgYzEyLjUtNywyNi43LTEwLjMsNDEtOS44YzQ4LjMsMCw4NC43LDQxLjcsODQuNywxMDMuM2MwLDczLjEtNDMuNywxMDkuMi05MSwxMDkuMmMtMTIuMSwwLjUtMjQuMi0yLjItMzUtNy44IGMtMTAuOC01LjYtMTkuOS0xMy45LTI2LjYtMjQuMmgtMC44VjI5MWgtMzZ2LTIyMEwtNDc4LjYsNzEuNEwtNDc4LjYsNzEuNHogTS00NDIuNiwxMjUuNmMwLjEsNS4xLDAuNiwxMC4xLDEuNywxNS4xIGMzLDEyLjMsOS45LDIzLjMsMTkuOCwzMS4xYzkuOSw3LjgsMjIuMSwxMi4xLDM0LjcsMTIuMWMzOC41LDAsNjAuNy0zMS45LDYwLjctNzguNWMwLTQwLjctMjEuMS03NS42LTU5LjUtNzUuNiBjLTEyLjksMC40LTI1LjMsNS4xLTM1LjMsMTMuNGMtOS45LDguMy0xNi45LDE5LjctMTkuNiwzMi40Yy0xLjUsNC45LTIuMywxMC0yLjUsMTUuMVYxMjUuNkwtNDQyLjYsMTI1LjZMLTQ0Mi42LDEyNS42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MDYuNzQwNzI2LCA1Ni44MzcxMDQpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC43NTEyMjYsIDEuOTg5Mjk5KSI+CiAgICAgICAgPHBhdGggZD0iTS00NDAuOCwwbDQzLjcsMTIwLjFjNC41LDEzLjQsOS41LDI5LjQsMTIuOCw0MS43aDAuOGMzLjctMTIuMiw3LjktMjcuNywxMi44LTQyLjQgbDM5LjctMTE5LjJoMzguNUwtMzQ2LjksMTQ1Yy0yNiw2OS43LTQzLjcsMTA1LjQtNjguNiwxMjcuMmMtMTIuNSwxMS43LTI3LjksMjAtNDQuNiwyMy45bC05LjEtMzEuMSBjMTEuNy0zLjksMjIuNS0xMC4xLDMxLjgtMTguMWMxMy4yLTExLjEsMjMuNy0yNS4yLDMwLjYtNDEuMmMxLjUtMi44LDIuNS01LjcsMi45LTguOGMtMC4zLTMuMy0xLjItNi42LTIuNS05LjdMLTQ4MC4yLDAuMSBoMzkuN0wtNDQwLjgsMEwtNDQwLjgsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODIyLjc0ODEwNCwgMC4wMDAwMDApIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS40NjQwNTAsIDAuMzc4OTE0KSI+CiAgICAgICAgPHBhdGggZD0iTS00MTMuNywwdjU4LjNoNTJ2MjguMmgtNTJWMTk2YzAsMjUsNywzOS41LDI3LjMsMzkuNWM3LjEsMC4xLDE0LjItMC43LDIxLjEtMi41IGwxLjcsMjcuN2MtMTAuMywzLjctMjEuMyw1LjQtMzIuMiw1Yy03LjMsMC40LTE0LjYtMC43LTIxLjMtMy40Yy02LjgtMi43LTEyLjktNi44LTE3LjktMTIuMWMtMTAuMy0xMC45LTE0LjEtMjktMTQuMS01Mi45IFY4Ni41aC0zMVY1OC4zaDMxVjkuNkwtNDEzLjcsMEwtNDEzLjcsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOTc0LjQzMzI4NiwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuOTkwMDM0LCAwLjYxMDMzOSkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDQ1LjgsMTEzYzAuOCw1MCwzMi4yLDcwLjYsNjguNiw3MC42YzE5LDAuNiwzNy45LTMsNTUuMy0xMC41bDYuMiwyNi40IGMtMjAuOSw4LjktNDMuNSwxMy4xLTY2LjIsMTIuNmMtNjEuNSwwLTk4LjMtNDEuMi05OC4zLTEwMi41Qy00ODAuMiw0OC4yLTQ0NC43LDAtMzg2LjUsMGM2NS4yLDAsODIuNyw1OC4zLDgyLjcsOTUuNyBjLTAuMSw1LjgtMC41LDExLjUtMS4yLDE3LjJoLTE0MC42SC00NDUuOEwtNDQ1LjgsMTEzeiBNLTMzOS4yLDg2LjZjMC40LTIzLjUtOS41LTYwLjEtNTAuNC02MC4xIGMtMzYuOCwwLTUyLjgsMzQuNC01NS43LDYwLjFILTMzOS4yTC0zMzkuMiw4Ni42TC0zMzkuMiw4Ni42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjAxLjk2MTA1OCwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMTc5NjQwLCAwLjcwNTA2OCkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDc4LjYsNjhjMC0yMy45LTAuNC00NC41LTEuNy02My40aDMxLjhsMS4yLDM5LjloMS43YzkuMS0yNy4zLDMxLTQ0LjUsNTUuMy00NC41IGMzLjUtMC4xLDcsMC40LDEwLjMsMS4ydjM0LjhjLTQuMS0wLjktOC4yLTEuMy0xMi40LTEuMmMtMjUuNiwwLTQzLjcsMTkuNy00OC43LDQ3LjRjLTEsNS43LTEuNiwxMS41LTEuNywxNy4ydjEwOC4zaC0zNlY2OCBMLTQ3OC42LDY4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCBkPSJNMTM1Mi4zLDMyNi4yaDM3VjI4aC0zN1YzMjYuMnogTTE2MDQuOCwzMjYuMmMtMi41LTEzLjktMy40LTMxLjEtMy40LTQ4Ljd2LTc2IGMwLTQwLjctMTUuMS04My4xLTc3LjMtODMuMWMtMjUuNiwwLTUwLDcuMS02Ni44LDE4LjFsOC40LDI0LjRjMTQuMy05LjIsMzQtMTUuMSw1My0xNS4xYzQxLjYsMCw0Ni4yLDMwLjIsNDYuMiw0N3Y0LjIgYy03OC42LTAuNC0xMjIuMywyNi41LTEyMi4zLDc1LjZjMCwyOS40LDIxLDU4LjQsNjIuMiw1OC40YzI5LDAsNTAuOS0xNC4zLDYyLjItMzAuMmgxLjNsMi45LDI1LjZIMTYwNC44eiBNMTU2NS43LDI1Ny43IGMwLDMuOC0wLjgsOC0yLjEsMTEuOGMtNS45LDE3LjItMjIuNywzNC00OS4yLDM0Yy0xOC45LDAtMzQuOS0xMS4zLTM0LjktMzUuM2MwLTM5LjUsNDUuOC00Ni42LDg2LjItNDUuOFYyNTcuN3ogTTE2OTguNSwzMjYuMiBsMS43LTMzLjZoMS4zYzE1LjEsMjYuOSwzOC43LDM4LjIsNjguMSwzOC4yYzQ1LjQsMCw5MS4yLTM2LjEsOTEuMi0xMDguOGMwLjQtNjEuNy0zNS4zLTEwMy43LTg1LjctMTAzLjcgYy0zMi44LDAtNTYuMywxNC43LTY5LjMsMzcuNGgtMC44VjI4aC0zNi42djI0NS43YzAsMTguMS0wLjgsMzguNi0xLjcsNTIuNUgxNjk4LjV6IE0xNzA0LjgsMjA4LjJjMC01LjksMS4zLTEwLjksMi4xLTE1LjEgYzcuNi0yOC4xLDMxLjEtNDUuNCw1Ni4zLTQ1LjRjMzkuNSwwLDYwLjUsMzQuOSw2MC41LDc1LjZjMCw0Ni42LTIzLjEsNzguMS02MS44LDc4LjFjLTI2LjksMC00OC4zLTE3LjYtNTUuNS00My4zIGMtMC44LTQuMi0xLjctOC44LTEuNy0xMy40VjIwOC4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzYxNjE2MSIgZD0iTTE1IDlIOXY2aDZWOXptLTIgNGgtMnYtMmgydjJ6bTgtMlY5aC0yVjdjMC0xLjEtLjktMi0yLTJoLTJWM2gtMnYyaC0yVjNIOXYySDdjLTEuMSAwLTIgLjktMiAydjJIM3YyaDJ2MkgzdjJoMnYyYzAgMS4xLjkgMiAyIDJoMnYyaDJ2LTJoMnYyaDJ2LTJoMmMxLjEgMCAyLS45IDItMnYtMmgydi0yaC0ydi0yaDJ6bS00IDZIN1Y3aDEwdjEweiIvPgo8L3N2Zz4K);
  --jp-icon-keyboard: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMTdjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0tOSAzaDJ2MmgtMlY4em0wIDNoMnYyaC0ydi0yek04IDhoMnYySDhWOHptMCAzaDJ2Mkg4di0yem0tMSAySDV2LTJoMnYyem0wLTNINVY4aDJ2MnptOSA3SDh2LTJoOHYyem0wLTRoLTJ2LTJoMnYyem0wLTNoLTJWOGgydjJ6bTMgM2gtMnYtMmgydjJ6bTAtM2gtMlY4aDJ2MnoiLz4KPC9zdmc+Cg==);
  --jp-icon-launcher: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkgMTlINVY1aDdWM0g1YTIgMiAwIDAwLTIgMnYxNGEyIDIgMCAwMDIgMmgxNGMxLjEgMCAyLS45IDItMnYtN2gtMnY3ek0xNCAzdjJoMy41OWwtOS44MyA5LjgzIDEuNDEgMS40MUwxOSA2LjQxVjEwaDJWM2gtN3oiLz4KPC9zdmc+Cg==);
  --jp-icon-line-form: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGZpbGw9IndoaXRlIiBkPSJNNS44OCA0LjEyTDEzLjc2IDEybC03Ljg4IDcuODhMOCAyMmwxMC0xMEw4IDJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-link: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMuOSAxMmMwLTEuNzEgMS4zOS0zLjEgMy4xLTMuMWg0VjdIN2MtMi43NiAwLTUgMi4yNC01IDVzMi4yNCA1IDUgNWg0di0xLjlIN2MtMS43MSAwLTMuMS0xLjM5LTMuMS0zLjF6TTggMTNoOHYtMkg4djJ6bTktNmgtNHYxLjloNGMxLjcxIDAgMy4xIDEuMzkgMy4xIDMuMXMtMS4zOSAzLjEtMy4xIDMuMWgtNFYxN2g0YzIuNzYgMCA1LTIuMjQgNS01cy0yLjI0LTUtNS01eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xOSA1djE0SDVWNWgxNG0xLjEtMkgzLjljLS41IDAtLjkuNC0uOS45djE2LjJjMCAuNC40LjkuOS45aDE2LjJjLjQgMCAuOS0uNS45LS45VjMuOWMwLS41LS41LS45LS45LS45ek0xMSA3aDZ2MmgtNlY3em0wIDRoNnYyaC02di0yem0wIDRoNnYyaC02ek03IDdoMnYySDd6bTAgNGgydjJIN3ptMCA0aDJ2Mkg3eiIvPgo8L3N2Zz4=);
  --jp-icon-listings-info: url(data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJDYXBhXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4Ig0KCSB2aWV3Qm94PSIwIDAgNTAuOTc4IDUwLjk3OCIgc3R5bGU9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgNTAuOTc4IDUwLjk3ODsiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxnPg0KCQkJPHBhdGggc3R5bGU9ImZpbGw6IzAxMDAwMjsiIGQ9Ik00My41Miw3LjQ1OEMzOC43MTEsMi42NDgsMzIuMzA3LDAsMjUuNDg5LDBDMTguNjcsMCwxMi4yNjYsMi42NDgsNy40NTgsNy40NTgNCgkJCQljLTkuOTQzLDkuOTQxLTkuOTQzLDI2LjExOSwwLDM2LjA2MmM0LjgwOSw0LjgwOSwxMS4yMTIsNy40NTYsMTguMDMxLDcuNDU4YzAsMCwwLjAwMSwwLDAuMDAyLDANCgkJCQljNi44MTYsMCwxMy4yMjEtMi42NDgsMTguMDI5LTcuNDU4YzQuODA5LTQuODA5LDcuNDU3LTExLjIxMiw3LjQ1Ny0xOC4wM0M1MC45NzcsMTguNjcsNDguMzI4LDEyLjI2Niw0My41Miw3LjQ1OHoNCgkJCQkgTTQyLjEwNiw0Mi4xMDVjLTQuNDMyLDQuNDMxLTEwLjMzMiw2Ljg3Mi0xNi42MTUsNi44NzJoLTAuMDAyYy02LjI4NS0wLjAwMS0xMi4xODctMi40NDEtMTYuNjE3LTYuODcyDQoJCQkJYy05LjE2Mi05LjE2My05LjE2Mi0yNC4wNzEsMC0zMy4yMzNDMTMuMzAzLDQuNDQsMTkuMjA0LDIsMjUuNDg5LDJjNi4yODQsMCwxMi4xODYsMi40NCwxNi42MTcsNi44NzINCgkJCQljNC40MzEsNC40MzEsNi44NzEsMTAuMzMyLDYuODcxLDE2LjYxN0M0OC45NzcsMzEuNzcyLDQ2LjUzNiwzNy42NzUsNDIuMTA2LDQyLjEwNXoiLz4NCgkJPC9nPg0KCQk8Zz4NCgkJCTxwYXRoIHN0eWxlPSJmaWxsOiMwMTAwMDI7IiBkPSJNMjMuNTc4LDMyLjIxOGMtMC4wMjMtMS43MzQsMC4xNDMtMy4wNTksMC40OTYtMy45NzJjMC4zNTMtMC45MTMsMS4xMS0xLjk5NywyLjI3Mi0zLjI1Mw0KCQkJCWMwLjQ2OC0wLjUzNiwwLjkyMy0xLjA2MiwxLjM2Ny0xLjU3NWMwLjYyNi0wLjc1MywxLjEwNC0xLjQ3OCwxLjQzNi0yLjE3NWMwLjMzMS0wLjcwNywwLjQ5NS0xLjU0MSwwLjQ5NS0yLjUNCgkJCQljMC0xLjA5Ni0wLjI2LTIuMDg4LTAuNzc5LTIuOTc5Yy0wLjU2NS0wLjg3OS0xLjUwMS0xLjMzNi0yLjgwNi0xLjM2OWMtMS44MDIsMC4wNTctMi45ODUsMC42NjctMy41NSwxLjgzMg0KCQkJCWMtMC4zMDEsMC41MzUtMC41MDMsMS4xNDEtMC42MDcsMS44MTRjLTAuMTM5LDAuNzA3LTAuMjA3LDEuNDMyLTAuMjA3LDIuMTc0aC0yLjkzN2MtMC4wOTEtMi4yMDgsMC40MDctNC4xMTQsMS40OTMtNS43MTkNCgkJCQljMS4wNjItMS42NCwyLjg1NS0yLjQ4MSw1LjM3OC0yLjUyN2MyLjE2LDAuMDIzLDMuODc0LDAuNjA4LDUuMTQxLDEuNzU4YzEuMjc4LDEuMTYsMS45MjksMi43NjQsMS45NSw0LjgxMQ0KCQkJCWMwLDEuMTQyLTAuMTM3LDIuMTExLTAuNDEsMi45MTFjLTAuMzA5LDAuODQ1LTAuNzMxLDEuNTkzLTEuMjY4LDIuMjQzYy0wLjQ5MiwwLjY1LTEuMDY4LDEuMzE4LTEuNzMsMi4wMDINCgkJCQljLTAuNjUsMC42OTctMS4zMTMsMS40NzktMS45ODcsMi4zNDZjLTAuMjM5LDAuMzc3LTAuNDI5LDAuNzc3LTAuNTY1LDEuMTk5Yy0wLjE2LDAuOTU5LTAuMjE3LDEuOTUxLTAuMTcxLDIuOTc5DQoJCQkJQzI2LjU4OSwzMi4yMTgsMjMuNTc4LDMyLjIxOCwyMy41NzgsMzIuMjE4eiBNMjMuNTc4LDM4LjIydi0zLjQ4NGgzLjA3NnYzLjQ4NEgyMy41Nzh6Ii8+DQoJCTwvZz4NCgk8L2c+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8L3N2Zz4NCg==);
  --jp-icon-markdown: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjN0IxRkEyIiBkPSJNNSAxNC45aDEybC02LjEgNnptOS40LTYuOGMwLTEuMy0uMS0yLjktLjEtNC41LS40IDEuNC0uOSAyLjktMS4zIDQuM2wtMS4zIDQuM2gtMkw4LjUgNy45Yy0uNC0xLjMtLjctMi45LTEtNC4zLS4xIDEuNi0uMSAzLjItLjIgNC42TDcgMTIuNEg0LjhsLjctMTFoMy4zTDEwIDVjLjQgMS4yLjcgMi43IDEgMy45LjMtMS4yLjctMi42IDEtMy45bDEuMi0zLjdoMy4zbC42IDExaC0yLjRsLS4zLTQuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-new-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDZoLThsLTItMkg0Yy0xLjExIDAtMS45OS44OS0xLjk5IDJMMiAxOGMwIDEuMTEuODkgMiAyIDJoMTZjMS4xMSAwIDItLjg5IDItMlY4YzAtMS4xMS0uODktMi0yLTJ6bS0xIDhoLTN2M2gtMnYtM2gtM3YtMmgzVjloMnYzaDN2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-not-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI1IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMTkgMTcuMTg0NCAyLjk2OTY4IDE0LjMwMzIgMS44NjA5NCAxMS40NDA5WiIvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24yIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOS4zMTU5MiA5LjMyMDMxKSIgZD0iTTcuMzY4NDIgMEwwIDcuMzY0NzkiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDkuMzE1OTIgMTYuNjgzNikgc2NhbGUoMSAtMSkiIGQ9Ik03LjM2ODQyIDBMMCA3LjM2NDc5Ii8+Cjwvc3ZnPgo=);
  --jp-icon-notebook: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNFRjZDMDAiPgogICAgPHBhdGggZD0iTTE4LjcgMy4zdjE1LjRIMy4zVjMuM2gxNS40bTEuNS0xLjVIMS44djE4LjNoMTguM2wuMS0xOC4zeiIvPgogICAgPHBhdGggZD0iTTE2LjUgMTYuNWwtNS40LTQuMy01LjYgNC4zdi0xMWgxMXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-numbering: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTQgMTlINlYxOS41SDVWMjAuNUg2VjIxSDRWMjJIN1YxOEg0VjE5Wk01IDEwSDZWNkg0VjdINVYxMFpNNCAxM0g1LjhMNCAxNS4xVjE2SDdWMTVINS4yTDcgMTIuOVYxMkg0VjEzWk05IDdWOUgyM1Y3SDlaTTkgMjFIMjNWMTlIOVYyMVpNOSAxNUgyM1YxM0g5VjE1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-offline-bolt: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDIuMDJjLTUuNTEgMC05Ljk4IDQuNDctOS45OCA5Ljk4czQuNDcgOS45OCA5Ljk4IDkuOTggOS45OC00LjQ3IDkuOTgtOS45OFMxNy41MSAyLjAyIDEyIDIuMDJ6TTExLjQ4IDIwdi02LjI2SDhMMTMgNHY2LjI2aDMuMzVMMTEuNDggMjB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-palette: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE4IDEzVjIwSDRWNkg5LjAyQzkuMDcgNS4yOSA5LjI0IDQuNjIgOS41IDRINEMyLjkgNCAyIDQuOSAyIDZWMjBDMiAyMS4xIDIuOSAyMiA0IDIySDE4QzE5LjEgMjIgMjAgMjEuMSAyMCAyMFYxNUwxOCAxM1pNMTkuMyA4Ljg5QzE5Ljc0IDguMTkgMjAgNy4zOCAyMCA2LjVDMjAgNC4wMSAxNy45OSAyIDE1LjUgMkMxMy4wMSAyIDExIDQuMDEgMTEgNi41QzExIDguOTkgMTMuMDEgMTEgMTUuNDkgMTFDMTYuMzcgMTEgMTcuMTkgMTAuNzQgMTcuODggMTAuM0wyMSAxMy40MkwyMi40MiAxMkwxOS4zIDguODlaTTE1LjUgOUMxNC4xMiA5IDEzIDcuODggMTMgNi41QzEzIDUuMTIgMTQuMTIgNCAxNS41IDRDMTYuODggNCAxOCA1LjEyIDE4IDYuNUMxOCA3Ljg4IDE2Ljg4IDkgMTUuNSA5WiIvPgogICAgPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik00IDZIOS4wMTg5NEM5LjAwNjM5IDYuMTY1MDIgOSA2LjMzMTc2IDkgNi41QzkgOC44MTU3NyAxMC4yMTEgMTAuODQ4NyAxMi4wMzQzIDEySDlWMTRIMTZWMTIuOTgxMUMxNi41NzAzIDEyLjkzNzcgMTcuMTIgMTIuODIwNyAxNy42Mzk2IDEyLjYzOTZMMTggMTNWMjBINFY2Wk04IDhINlYxMEg4VjhaTTYgMTJIOFYxNEg2VjEyWk04IDE2SDZWMThIOFYxNlpNOSAxNkgxNlYxOEg5VjE2WiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-paste: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE5IDJoLTQuMThDMTQuNC44NCAxMy4zIDAgMTIgMGMtMS4zIDAtMi40Ljg0LTIuODIgMkg1Yy0xLjEgMC0yIC45LTIgMnYxNmMwIDEuMS45IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjRjMC0xLjEtLjktMi0yLTJ6bS03IDBjLjU1IDAgMSAuNDUgMSAxcy0uNDUgMS0xIDEtMS0uNDUtMS0xIC40NS0xIDEtMXptNyAxOEg1VjRoMnYzaDEwVjRoMnYxNnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-pdf: url(data:image/svg+xml;base64,PHN2ZwogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMiAyMiIgd2lkdGg9IjE2Ij4KICAgIDxwYXRoIHRyYW5zZm9ybT0icm90YXRlKDQ1KSIgY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0ZGMkEyQSIKICAgICAgIGQ9Im0gMjIuMzQ0MzY5LC0zLjAxNjM2NDIgaCA1LjYzODYwNCB2IDEuNTc5MjQzMyBoIC0zLjU0OTIyNyB2IDEuNTA4NjkyOTkgaCAzLjMzNzU3NiBWIDEuNjUwODE1NCBoIC0zLjMzNzU3NiB2IDMuNDM1MjYxMyBoIC0yLjA4OTM3NyB6IG0gLTcuMTM2NDQ0LDEuNTc5MjQzMyB2IDQuOTQzOTU0MyBoIDAuNzQ4OTIgcSAxLjI4MDc2MSwwIDEuOTUzNzAzLC0wLjYzNDk1MzUgMC42NzgzNjksLTAuNjM0OTUzNSAwLjY3ODM2OSwtMS44NDUxNjQxIDAsLTEuMjA0NzgzNTUgLTAuNjcyOTQyLC0xLjgzNDMxMDExIC0wLjY3Mjk0MiwtMC42Mjk1MjY1OSAtMS45NTkxMywtMC42Mjk1MjY1OSB6IG0gLTIuMDg5Mzc3LC0xLjU3OTI0MzMgaCAyLjIwMzM0MyBxIDEuODQ1MTY0LDAgMi43NDYwMzksMC4yNjU5MjA3IDAuOTA2MzAxLDAuMjYwNDkzNyAxLjU1MjEwOCwwLjg5MDAyMDMgMC41Njk4MywwLjU0ODEyMjMgMC44NDY2MDUsMS4yNjQ0ODAwNiAwLjI3Njc3NCwwLjcxNjM1NzgxIDAuMjc2Nzc0LDEuNjIyNjU4OTQgMCwwLjkxNzE1NTEgLTAuMjc2Nzc0LDEuNjM4OTM5OSAtMC4yNzY3NzUsMC43MTYzNTc4IC0wLjg0NjYwNSwxLjI2NDQ4IC0wLjY1MTIzNCwwLjYyOTUyNjYgLTEuNTYyOTYyLDAuODk1NDQ3MyAtMC45MTE3MjgsMC4yNjA0OTM3IC0yLjczNTE4NSwwLjI2MDQ5MzcgaCAtMi4yMDMzNDMgeiBtIC04LjE0NTg1NjUsMCBoIDMuNDY3ODIzIHEgMS41NDY2ODE2LDAgMi4zNzE1Nzg1LDAuNjg5MjIzIDAuODMwMzI0LDAuNjgzNzk2MSAwLjgzMDMyNCwxLjk1MzcwMzE0IDAsMS4yNzUzMzM5NyAtMC44MzAzMjQsMS45NjQ1NTcwNiBRIDkuOTg3MTk2MSwyLjI3NDkxNSA4LjQ0MDUxNDUsMi4yNzQ5MTUgSCA3LjA2MjA2ODQgViA1LjA4NjA3NjcgSCA0Ljk3MjY5MTUgWiBtIDIuMDg5Mzc2OSwxLjUxNDExOTkgdiAyLjI2MzAzOTQzIGggMS4xNTU5NDEgcSAwLjYwNzgxODgsMCAwLjkzODg2MjksLTAuMjkzMDU1NDcgMC4zMzEwNDQxLC0wLjI5ODQ4MjQxIDAuMzMxMDQ0MSwtMC44NDExNzc3MiAwLC0wLjU0MjY5NTMxIC0wLjMzMTA0NDEsLTAuODM1NzUwNzQgLTAuMzMxMDQ0MSwtMC4yOTMwNTU1IC0wLjkzODg2MjksLTAuMjkzMDU1NSB6IgovPgo8L3N2Zz4K);
  --jp-icon-python: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMEQ0N0ExIj4KICAgIDxwYXRoIGQ9Ik0xMS4xIDYuOVY1LjhINi45YzAtLjUgMC0xLjMuMi0xLjYuNC0uNy44LTEuMSAxLjctMS40IDEuNy0uMyAyLjUtLjMgMy45LS4xIDEgLjEgMS45LjkgMS45IDEuOXY0LjJjMCAuNS0uOSAxLjYtMiAxLjZIOC44Yy0xLjUgMC0yLjQgMS40LTIuNCAyLjh2Mi4ySDQuN0MzLjUgMTUuMSAzIDE0IDMgMTMuMVY5Yy0uMS0xIC42LTIgMS44LTIgMS41LS4xIDYuMy0uMSA2LjMtLjF6Ii8+CiAgICA8cGF0aCBkPSJNMTAuOSAxNS4xdjEuMWg0LjJjMCAuNSAwIDEuMy0uMiAxLjYtLjQuNy0uOCAxLjEtMS43IDEuNC0xLjcuMy0yLjUuMy0zLjkuMS0xLS4xLTEuOS0uOS0xLjktMS45di00LjJjMC0uNS45LTEuNiAyLTEuNmgzLjhjMS41IDAgMi40LTEuNCAyLjQtMi44VjYuNmgxLjdDMTguNSA2LjkgMTkgOCAxOSA4LjlWMTNjMCAxLS43IDIuMS0xLjkgMi4xaC02LjJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-r-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjE5NkYzIiBkPSJNNC40IDIuNWMxLjItLjEgMi45LS4zIDQuOS0uMyAyLjUgMCA0LjEuNCA1LjIgMS4zIDEgLjcgMS41IDEuOSAxLjUgMy41IDAgMi0xLjQgMy41LTIuOSA0LjEgMS4yLjQgMS43IDEuNiAyLjIgMyAuNiAxLjkgMSAzLjkgMS4zIDQuNmgtMy44Yy0uMy0uNC0uOC0xLjctMS4yLTMuN3MtMS4yLTIuNi0yLjYtMi42aC0uOXY2LjRINC40VjIuNXptMy43IDYuOWgxLjRjMS45IDAgMi45LS45IDIuOS0yLjNzLTEtMi4zLTIuOC0yLjNjLS43IDAtMS4zIDAtMS42LjJ2NC41aC4xdi0uMXoiLz4KPC9zdmc+Cg==);
  --jp-icon-react: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMTUwIDE1MCA1NDEuOSAyOTUuMyI+CiAgPGcgY2xhc3M9ImpwLWljb24tYnJhbmQyIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxREFGQiI+CiAgICA8cGF0aCBkPSJNNjY2LjMgMjk2LjVjMC0zMi41LTQwLjctNjMuMy0xMDMuMS04Mi40IDE0LjQtNjMuNiA4LTExNC4yLTIwLjItMTMwLjQtNi41LTMuOC0xNC4xLTUuNi0yMi40LTUuNnYyMi4zYzQuNiAwIDguMy45IDExLjQgMi42IDEzLjYgNy44IDE5LjUgMzcuNSAxNC45IDc1LjctMS4xIDkuNC0yLjkgMTkuMy01LjEgMjkuNC0xOS42LTQuOC00MS04LjUtNjMuNS0xMC45LTEzLjUtMTguNS0yNy41LTM1LjMtNDEuNi01MCAzMi42LTMwLjMgNjMuMi00Ni45IDg0LTQ2LjlWNzhjLTI3LjUgMC02My41IDE5LjYtOTkuOSA1My42LTM2LjQtMzMuOC03Mi40LTUzLjItOTkuOS01My4ydjIyLjNjMjAuNyAwIDUxLjQgMTYuNSA4NCA0Ni42LTE0IDE0LjctMjggMzEuNC00MS4zIDQ5LjktMjIuNiAyLjQtNDQgNi4xLTYzLjYgMTEtMi4zLTEwLTQtMTkuNy01LjItMjktNC43LTM4LjIgMS4xLTY3LjkgMTQuNi03NS44IDMtMS44IDYuOS0yLjYgMTEuNS0yLjZWNzguNWMtOC40IDAtMTYgMS44LTIyLjYgNS42LTI4LjEgMTYuMi0zNC40IDY2LjctMTkuOSAxMzAuMS02Mi4yIDE5LjItMTAyLjcgNDkuOS0xMDIuNyA4Mi4zIDAgMzIuNSA0MC43IDYzLjMgMTAzLjEgODIuNC0xNC40IDYzLjYtOCAxMTQuMiAyMC4yIDEzMC40IDYuNSAzLjggMTQuMSA1LjYgMjIuNSA1LjYgMjcuNSAwIDYzLjUtMTkuNiA5OS45LTUzLjYgMzYuNCAzMy44IDcyLjQgNTMuMiA5OS45IDUzLjIgOC40IDAgMTYtMS44IDIyLjYtNS42IDI4LjEtMTYuMiAzNC40LTY2LjcgMTkuOS0xMzAuMSA2Mi0xOS4xIDEwMi41LTQ5LjkgMTAyLjUtODIuM3ptLTEzMC4yLTY2LjdjLTMuNyAxMi45LTguMyAyNi4yLTEzLjUgMzkuNS00LjEtOC04LjQtMTYtMTMuMS0yNC00LjYtOC05LjUtMTUuOC0xNC40LTIzLjQgMTQuMiAyLjEgMjcuOSA0LjcgNDEgNy45em0tNDUuOCAxMDYuNWMtNy44IDEzLjUtMTUuOCAyNi4zLTI0LjEgMzguMi0xNC45IDEuMy0zMCAyLTQ1LjIgMi0xNS4xIDAtMzAuMi0uNy00NS0xLjktOC4zLTExLjktMTYuNC0yNC42LTI0LjItMzgtNy42LTEzLjEtMTQuNS0yNi40LTIwLjgtMzkuOCA2LjItMTMuNCAxMy4yLTI2LjggMjAuNy0zOS45IDcuOC0xMy41IDE1LjgtMjYuMyAyNC4xLTM4LjIgMTQuOS0xLjMgMzAtMiA0NS4yLTIgMTUuMSAwIDMwLjIuNyA0NSAxLjkgOC4zIDExLjkgMTYuNCAyNC42IDI0LjIgMzggNy42IDEzLjEgMTQuNSAyNi40IDIwLjggMzkuOC02LjMgMTMuNC0xMy4yIDI2LjgtMjAuNyAzOS45em0zMi4zLTEzYzUuNCAxMy40IDEwIDI2LjggMTMuOCAzOS44LTEzLjEgMy4yLTI2LjkgNS45LTQxLjIgOCA0LjktNy43IDkuOC0xNS42IDE0LjQtMjMuNyA0LjYtOCA4LjktMTYuMSAxMy0yNC4xek00MjEuMiA0MzBjLTkuMy05LjYtMTguNi0yMC4zLTI3LjgtMzIgOSAuNCAxOC4yLjcgMjcuNS43IDkuNCAwIDE4LjctLjIgMjcuOC0uNy05IDExLjctMTguMyAyMi40LTI3LjUgMzJ6bS03NC40LTU4LjljLTE0LjItMi4xLTI3LjktNC43LTQxLTcuOSAzLjctMTIuOSA4LjMtMjYuMiAxMy41LTM5LjUgNC4xIDggOC40IDE2IDEzLjEgMjQgNC43IDggOS41IDE1LjggMTQuNCAyMy40ek00MjAuNyAxNjNjOS4zIDkuNiAxOC42IDIwLjMgMjcuOCAzMi05LS40LTE4LjItLjctMjcuNS0uNy05LjQgMC0xOC43LjItMjcuOC43IDktMTEuNyAxOC4zLTIyLjQgMjcuNS0zMnptLTc0IDU4LjljLTQuOSA3LjctOS44IDE1LjYtMTQuNCAyMy43LTQuNiA4LTguOSAxNi0xMyAyNC01LjQtMTMuNC0xMC0yNi44LTEzLjgtMzkuOCAxMy4xLTMuMSAyNi45LTUuOCA0MS4yLTcuOXptLTkwLjUgMTI1LjJjLTM1LjQtMTUuMS01OC4zLTM0LjktNTguMy01MC42IDAtMTUuNyAyMi45LTM1LjYgNTguMy01MC42IDguNi0zLjcgMTgtNyAyNy43LTEwLjEgNS43IDE5LjYgMTMuMiA0MCAyMi41IDYwLjktOS4yIDIwLjgtMTYuNiA0MS4xLTIyLjIgNjAuNi05LjktMy4xLTE5LjMtNi41LTI4LTEwLjJ6TTMxMCA0OTBjLTEzLjYtNy44LTE5LjUtMzcuNS0xNC45LTc1LjcgMS4xLTkuNCAyLjktMTkuMyA1LjEtMjkuNCAxOS42IDQuOCA0MSA4LjUgNjMuNSAxMC45IDEzLjUgMTguNSAyNy41IDM1LjMgNDEuNiA1MC0zMi42IDMwLjMtNjMuMiA0Ni45LTg0IDQ2LjktNC41LS4xLTguMy0xLTExLjMtMi43em0yMzcuMi03Ni4yYzQuNyAzOC4yLTEuMSA2Ny45LTE0LjYgNzUuOC0zIDEuOC02LjkgMi42LTExLjUgMi42LTIwLjcgMC01MS40LTE2LjUtODQtNDYuNiAxNC0xNC43IDI4LTMxLjQgNDEuMy00OS45IDIyLjYtMi40IDQ0LTYuMSA2My42LTExIDIuMyAxMC4xIDQuMSAxOS44IDUuMiAyOS4xem0zOC41LTY2LjdjLTguNiAzLjctMTggNy0yNy43IDEwLjEtNS43LTE5LjYtMTMuMi00MC0yMi41LTYwLjkgOS4yLTIwLjggMTYuNi00MS4xIDIyLjItNjAuNiA5LjkgMy4xIDE5LjMgNi41IDI4LjEgMTAuMiAzNS40IDE1LjEgNTguMyAzNC45IDU4LjMgNTAuNi0uMSAxNS43LTIzIDM1LjYtNTguNCA1MC42ek0zMjAuOCA3OC40eiIvPgogICAgPGNpcmNsZSBjeD0iNDIwLjkiIGN5PSIyOTYuNSIgcj0iNDUuNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-redo: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTE4LjQgMTAuNkMxNi41NSA4Ljk5IDE0LjE1IDggMTEuNSA4Yy00LjY1IDAtOC41OCAzLjAzLTkuOTYgNy4yMkwzLjkgMTZjMS4wNS0zLjE5IDQuMDUtNS41IDcuNi01LjUgMS45NSAwIDMuNzMuNzIgNS4xMiAxLjg4TDEzIDE2aDlWN2wtMy42IDMuNnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-refresh: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTkgMTMuNWMtMi40OSAwLTQuNS0yLjAxLTQuNS00LjVTNi41MSA0LjUgOSA0LjVjMS4yNCAwIDIuMzYuNTIgMy4xNyAxLjMzTDEwIDhoNVYzbC0xLjc2IDEuNzZDMTIuMTUgMy42OCAxMC42NiAzIDkgMyA1LjY5IDMgMy4wMSA1LjY5IDMuMDEgOVM1LjY5IDE1IDkgMTVjMi45NyAwIDUuNDMtMi4xNiA1LjktNWgtMS41MmMtLjQ2IDItMi4yNCAzLjUtNC4zOCAzLjV6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-regex: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiBmaWxsPSIjRkZGIj4KICAgIDxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjUuNSIgY3k9IjE0LjUiIHI9IjEuNSIvPgogICAgPHJlY3QgeD0iMTIiIHk9IjQiIGNsYXNzPSJzdDIiIHdpZHRoPSIxIiBoZWlnaHQ9IjgiLz4KICAgIDxyZWN0IHg9IjguNSIgeT0iNy41IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjg2NiAtMC41IDAuNSAwLjg2NiAtMi4zMjU1IDcuMzIxOSkiIGNsYXNzPSJzdDIiIHdpZHRoPSI4IiBoZWlnaHQ9IjEiLz4KICAgIDxyZWN0IHg9IjEyIiB5PSI0IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjUgLTAuODY2IDAuODY2IDAuNSAtMC42Nzc5IDE0LjgyNTIpIiBjbGFzcz0ic3QyIiB3aWR0aD0iMSIgaGVpZ2h0PSI4Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-run: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTggNXYxNGwxMS03eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-running: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMjU2IDhDMTE5IDggOCAxMTkgOCAyNTZzMTExIDI0OCAyNDggMjQ4IDI0OC0xMTEgMjQ4LTI0OFMzOTMgOCAyNTYgOHptOTYgMzI4YzAgOC44LTcuMiAxNi0xNiAxNkgxNzZjLTguOCAwLTE2LTcuMi0xNi0xNlYxNzZjMC04LjggNy4yLTE2IDE2LTE2aDE2MGM4LjggMCAxNiA3LjIgMTYgMTZ2MTYweiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-save: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE3IDNINWMtMS4xMSAwLTIgLjktMiAydjE0YzAgMS4xLjg5IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjdsLTQtNHptLTUgMTZjLTEuNjYgMC0zLTEuMzQtMy0zczEuMzQtMyAzLTMgMyAxLjM0IDMgMy0xLjM0IDMtMyAzem0zLTEwSDVWNWgxMHY0eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-search: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-settings: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuNDMgMTIuOThjLjA0LS4zMi4wNy0uNjQuMDctLjk4cy0uMDMtLjY2LS4wNy0uOThsMi4xMS0xLjY1Yy4xOS0uMTUuMjQtLjQyLjEyLS42NGwtMi0zLjQ2Yy0uMTItLjIyLS4zOS0uMy0uNjEtLjIybC0yLjQ5IDFjLS41Mi0uNC0xLjA4LS43My0xLjY5LS45OGwtLjM4LTIuNjVBLjQ4OC40ODggMCAwMDE0IDJoLTRjLS4yNSAwLS40Ni4xOC0uNDkuNDJsLS4zOCAyLjY1Yy0uNjEuMjUtMS4xNy41OS0xLjY5Ljk4bC0yLjQ5LTFjLS4yMy0uMDktLjQ5IDAtLjYxLjIybC0yIDMuNDZjLS4xMy4yMi0uMDcuNDkuMTIuNjRsMi4xMSAxLjY1Yy0uMDQuMzItLjA3LjY1LS4wNy45OHMuMDMuNjYuMDcuOThsLTIuMTEgMS42NWMtLjE5LjE1LS4yNC40Mi0uMTIuNjRsMiAzLjQ2Yy4xMi4yMi4zOS4zLjYxLjIybDIuNDktMWMuNTIuNCAxLjA4LjczIDEuNjkuOThsLjM4IDIuNjVjLjAzLjI0LjI0LjQyLjQ5LjQyaDRjLjI1IDAgLjQ2LS4xOC40OS0uNDJsLjM4LTIuNjVjLjYxLS4yNSAxLjE3LS41OSAxLjY5LS45OGwyLjQ5IDFjLjIzLjA5LjQ5IDAgLjYxLS4yMmwyLTMuNDZjLjEyLS4yMi4wNy0uNDktLjEyLS42NGwtMi4xMS0xLjY1ek0xMiAxNS41Yy0xLjkzIDAtMy41LTEuNTctMy41LTMuNXMxLjU3LTMuNSAzLjUtMy41IDMuNSAxLjU3IDMuNSAzLjUtMS41NyAzLjUtMy41IDMuNXoiLz4KPC9zdmc+Cg==);
  --jp-icon-spreadsheet: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNENBRjUwIiBkPSJNMi4yIDIuMnYxNy42aDE3LjZWMi4ySDIuMnptMTUuNCA3LjdoLTUuNVY0LjRoNS41djUuNXpNOS45IDQuNHY1LjVINC40VjQuNGg1LjV6bS01LjUgNy43aDUuNXY1LjVINC40di01LjV6bTcuNyA1LjV2LTUuNWg1LjV2NS41aC01LjV6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-stop: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik02IDZoMTJ2MTJINnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tab: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIxIDNIM2MtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxOGMxLjEgMCAyLS45IDItMlY1YzAtMS4xLS45LTItMi0yem0wIDE2SDNWNWgxMHY0aDh2MTB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-table-rows: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMSw4SDNWNGgxOFY4eiBNMjEsMTBIM3Y0aDE4VjEweiBNMjEsMTZIM3Y0aDE4VjE2eiIvPgogICAgPC9nPgo8L3N2Zz4=);
  --jp-icon-tag: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCA0MyAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTI4LjgzMzIgMTIuMzM0TDMyLjk5OTggMTYuNTAwN0wzNy4xNjY1IDEyLjMzNEgyOC44MzMyWiIvPgoJCTxwYXRoIGQ9Ik0xNi4yMDk1IDIxLjYxMDRDMTUuNjg3MyAyMi4xMjk5IDE0Ljg0NDMgMjIuMTI5OSAxNC4zMjQ4IDIxLjYxMDRMNi45ODI5IDE0LjcyNDVDNi41NzI0IDE0LjMzOTQgNi4wODMxMyAxMy42MDk4IDYuMDQ3ODYgMTMuMDQ4MkM1Ljk1MzQ3IDExLjUyODggNi4wMjAwMiA4LjYxOTQ0IDYuMDY2MjEgNy4wNzY5NUM2LjA4MjgxIDYuNTE0NzcgNi41NTU0OCA2LjA0MzQ3IDcuMTE4MDQgNi4wMzA1NUM5LjA4ODYzIDUuOTg0NzMgMTMuMjYzOCA1LjkzNTc5IDEzLjY1MTggNi4zMjQyNUwyMS43MzY5IDEzLjYzOUMyMi4yNTYgMTQuMTU4NSAyMS43ODUxIDE1LjQ3MjQgMjEuMjYyIDE1Ljk5NDZMMTYuMjA5NSAyMS42MTA0Wk05Ljc3NTg1IDguMjY1QzkuMzM1NTEgNy44MjU2NiA4LjYyMzUxIDcuODI1NjYgOC4xODI4IDguMjY1QzcuNzQzNDYgOC43MDU3MSA3Ljc0MzQ2IDkuNDE3MzMgOC4xODI4IDkuODU2NjdDOC42MjM4MiAxMC4yOTY0IDkuMzM1ODIgMTAuMjk2NCA5Ljc3NTg1IDkuODU2NjdDMTAuMjE1NiA5LjQxNzMzIDEwLjIxNTYgOC43MDUzMyA5Ljc3NTg1IDguMjY1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-terminal: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiA+CiAgICA8cmVjdCBjbGFzcz0ianAtaWNvbjIganAtaWNvbi1zZWxlY3RhYmxlIiB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMikiIGZpbGw9IiMzMzMzMzMiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uLWFjY2VudDIganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGQ9Ik01LjA1NjY0IDguNzYxNzJDNS4wNTY2NCA4LjU5NzY2IDUuMDMxMjUgOC40NTMxMiA0Ljk4MDQ3IDguMzI4MTJDNC45MzM1OSA4LjE5OTIyIDQuODU1NDcgOC4wODIwMyA0Ljc0NjA5IDcuOTc2NTZDNC42NDA2MiA3Ljg3MTA5IDQuNSA3Ljc3NTM5IDQuMzI0MjIgNy42ODk0NUM0LjE1MjM0IDcuNTk5NjEgMy45NDMzNiA3LjUxMTcyIDMuNjk3MjcgNy40MjU3OEMzLjMwMjczIDcuMjg1MTYgMi45NDMzNiA3LjEzNjcyIDIuNjE5MTQgNi45ODA0N0MyLjI5NDkyIDYuODI0MjIgMi4wMTc1OCA2LjY0MjU4IDEuNzg3MTEgNi40MzU1NUMxLjU2MDU1IDYuMjI4NTIgMS4zODQ3NyA1Ljk4ODI4IDEuMjU5NzcgNS43MTQ4NEMxLjEzNDc3IDUuNDM3NSAxLjA3MjI3IDUuMTA5MzggMS4wNzIyNyA0LjczMDQ3QzEuMDcyMjcgNC4zOTg0NCAxLjEyODkxIDQuMDk1NyAxLjI0MjE5IDMuODIyMjdDMS4zNTU0NyAzLjU0NDkyIDEuNTE1NjIgMy4zMDQ2OSAxLjcyMjY2IDMuMTAxNTZDMS45Mjk2OSAyLjg5ODQ0IDIuMTc5NjkgMi43MzQzNyAyLjQ3MjY2IDIuNjA5MzhDMi43NjU2MiAyLjQ4NDM4IDMuMDkxOCAyLjQwNDMgMy40NTExNyAyLjM2OTE0VjEuMTA5MzhINC4zODg2N1YyLjM4MDg2QzQuNzQwMjMgMi40Mjc3MyA1LjA1NjY0IDIuNTIzNDQgNS4zMzc4OSAyLjY2Nzk3QzUuNjE5MTQgMi44MTI1IDUuODU3NDIgMy4wMDE5NSA2LjA1MjczIDMuMjM2MzNDNi4yNTE5NSAzLjQ2NjggNi40MDQzIDMuNzQwMjMgNi41MDk3NyA0LjA1NjY0QzYuNjE5MTQgNC4zNjkxNCA2LjY3MzgzIDQuNzIwNyA2LjY3MzgzIDUuMTExMzNINS4wNDQ5MkM1LjA0NDkyIDQuNjM4NjcgNC45Mzc1IDQuMjgxMjUgNC43MjI2NiA0LjAzOTA2QzQuNTA3ODEgMy43OTI5NyA0LjIxNjggMy42Njk5MiAzLjg0OTYxIDMuNjY5OTJDMy42NTAzOSAzLjY2OTkyIDMuNDc2NTYgMy42OTcyNyAzLjMyODEyIDMuNzUxOTVDMy4xODM1OSAzLjgwMjczIDMuMDY0NDUgMy44NzY5NSAyLjk3MDcgMy45NzQ2MUMyLjg3Njk1IDQuMDY4MzYgMi44MDY2NCA0LjE3OTY5IDIuNzU5NzcgNC4zMDg1OUMyLjcxNjggNC40Mzc1IDIuNjk1MzEgNC41NzgxMiAyLjY5NTMxIDQuNzMwNDdDMi42OTUzMSA0Ljg4MjgxIDIuNzE2OCA1LjAxOTUzIDIuNzU5NzcgNS4xNDA2MkMyLjgwNjY0IDUuMjU3ODEgMi44ODI4MSA1LjM2NzE5IDIuOTg4MjggNS40Njg3NUMzLjA5NzY2IDUuNTcwMzEgMy4yNDAyMyA1LjY2Nzk3IDMuNDE2MDIgNS43NjE3MkMzLjU5MTggNS44NTE1NiAzLjgxMDU1IDUuOTQzMzYgNC4wNzIyNyA2LjAzNzExQzQuNDY2OCA2LjE4NTU1IDQuODI0MjIgNi4zMzk4NCA1LjE0NDUzIDYuNUM1LjQ2NDg0IDYuNjU2MjUgNS43MzgyOCA2LjgzOTg0IDUuOTY0ODQgNy4wNTA3OEM2LjE5NTMxIDcuMjU3ODEgNi4zNzEwOSA3LjUgNi40OTIxOSA3Ljc3NzM0QzYuNjE3MTkgOC4wNTA3OCA2LjY3OTY5IDguMzc1IDYuNjc5NjkgOC43NUM2LjY3OTY5IDkuMDkzNzUgNi42MjMwNSA5LjQwNDMgNi41MDk3NyA5LjY4MTY0QzYuMzk2NDggOS45NTUwOCA2LjIzNDM4IDEwLjE5MTQgNi4wMjM0NCAxMC4zOTA2QzUuODEyNSAxMC41ODk4IDUuNTU4NTkgMTAuNzUgNS4yNjE3MiAxMC44NzExQzQuOTY0ODQgMTAuOTg4MyA0LjYzMjgxIDExLjA2NDUgNC4yNjU2MiAxMS4wOTk2VjEyLjI0OEgzLjMzMzk4VjExLjA5OTZDMy4wMDE5NSAxMS4wNjg0IDIuNjc5NjkgMTAuOTk2MSAyLjM2NzE5IDEwLjg4MjhDMi4wNTQ2OSAxMC43NjU2IDEuNzc3MzQgMTAuNTk3NyAxLjUzNTE2IDEwLjM3ODlDMS4yOTY4OCAxMC4xNjAyIDEuMTA1NDcgOS44ODQ3NyAwLjk2MDkzOCA5LjU1MjczQzAuODE2NDA2IDkuMjE2OCAwLjc0NDE0MSA4LjgxNDQ1IDAuNzQ0MTQxIDguMzQ1N0gyLjM3ODkxQzIuMzc4OTEgOC42MjY5NSAyLjQxOTkyIDguODYzMjggMi41MDE5NSA5LjA1NDY5QzIuNTgzOTggOS4yNDIxOSAyLjY4OTQ1IDkuMzkyNTggMi44MTgzNiA5LjUwNTg2QzIuOTUxMTcgOS42MTUyMyAzLjEwMTU2IDkuNjkzMzYgMy4yNjk1MyA5Ljc0MDIzQzMuNDM3NSA5Ljc4NzExIDMuNjA5MzggOS44MTA1NSAzLjc4NTE2IDkuODEwNTVDNC4yMDMxMiA5LjgxMDU1IDQuNTE5NTMgOS43MTI4OSA0LjczNDM4IDkuNTE3NThDNC45NDkyMiA5LjMyMjI3IDUuMDU2NjQgOS4wNzAzMSA1LjA1NjY0IDguNzYxNzJaTTEzLjQxOCAxMi4yNzE1SDguMDc0MjJWMTFIMTMuNDE4VjEyLjI3MTVaIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzLjk1MjY0IDYpIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K);
  --jp-icon-text-editor: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTUgMTVIM3YyaDEydi0yem0wLThIM3YyaDEyVjd6TTMgMTNoMTh2LTJIM3Yyem0wIDhoMTh2LTJIM3Yyek0zIDN2MmgxOFYzSDN6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-toc: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB2ZXJzaW9uPSIxLjEiIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgoJPHBhdGggZD0iTTcsNUgyMVY3SDdWNU03LDEzVjExSDIxVjEzSDdNNCw0LjVBMS41LDEuNSAwIDAsMSA1LjUsNkExLjUsMS41IDAgMCwxIDQsNy41QTEuNSwxLjUgMCAwLDEgMi41LDZBMS41LDEuNSAwIDAsMSA0LDQuNU00LDEwLjVBMS41LDEuNSAwIDAsMSA1LjUsMTJBMS41LDEuNSAwIDAsMSA0LDEzLjVBMS41LDEuNSAwIDAsMSAyLjUsMTJBMS41LDEuNSAwIDAsMSA0LDEwLjVNNywxOVYxN0gyMVYxOUg3TTQsMTYuNUExLjUsMS41IDAgMCwxIDUuNSwxOEExLjUsMS41IDAgMCwxIDQsMTkuNUExLjUsMS41IDAgMCwxIDIuNSwxOEExLjUsMS41IDAgMCwxIDQsMTYuNVoiIC8+Cjwvc3ZnPgo=);
  --jp-icon-tree-view: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMiAxMVYzaC03djNIOVYzSDJ2OGg3VjhoMnYxMGg0djNoN3YtOGgtN3YzaC0yVjhoMnYzeiIvPgogICAgPC9nPgo8L3N2Zz4=);
  --jp-icon-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMiAxNy4xODQ0IDIuOTY5NjggMTQuMzAzMiAxLjg2MDk0IDExLjQ0MDlaIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiMzMzMzMzMiIHN0cm9rZT0iIzMzMzMzMyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOCA5Ljg2NzE5KSIgZD0iTTIuODYwMTUgNC44NjUzNUwwLjcyNjU0OSAyLjk5OTU5TDAgMy42MzA0NUwyLjg2MDE1IDYuMTMxNTdMOCAwLjYzMDg3Mkw3LjI3ODU3IDBMMi44NjAxNSA0Ljg2NTM1WiIvPgo8L3N2Zz4K);
  --jp-icon-undo: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjUgOGMtMi42NSAwLTUuMDUuOTktNi45IDIuNkwyIDd2OWg5bC0zLjYyLTMuNjJjMS4zOS0xLjE2IDMuMTYtMS44OCA1LjEyLTEuODggMy41NCAwIDYuNTUgMi4zMSA3LjYgNS41bDIuMzctLjc4QzIxLjA4IDExLjAzIDE3LjE1IDggMTIuNSA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-vega: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbjEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjEyMTIxIj4KICAgIDxwYXRoIGQ9Ik0xMC42IDUuNGwyLjItMy4ySDIuMnY3LjNsNC02LjZ6Ii8+CiAgICA8cGF0aCBkPSJNMTUuOCAyLjJsLTQuNCA2LjZMNyA2LjNsLTQuOCA4djUuNWgxNy42VjIuMmgtNHptLTcgMTUuNEg1LjV2LTQuNGgzLjN2NC40em00LjQgMEg5LjhWOS44aDMuNHY3Ljh6bTQuNCAwaC0zLjRWNi41aDMuNHYxMS4xeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-yaml: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1jb250cmFzdDIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjRDgxQjYwIj4KICAgIDxwYXRoIGQ9Ik03LjIgMTguNnYtNS40TDMgNS42aDMuM2wxLjQgMy4xYy4zLjkuNiAxLjYgMSAyLjUuMy0uOC42LTEuNiAxLTIuNWwxLjQtMy4xaDMuNGwtNC40IDcuNnY1LjVsLTIuOS0uMXoiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxNi41IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxMSIgcj0iMi4xIi8+CiAgPC9nPgo8L3N2Zz4K);
}

/* Icon CSS class declarations */

.jp-AddIcon {
  background-image: var(--jp-icon-add);
}
.jp-BugIcon {
  background-image: var(--jp-icon-bug);
}
.jp-BuildIcon {
  background-image: var(--jp-icon-build);
}
.jp-CaretDownEmptyIcon {
  background-image: var(--jp-icon-caret-down-empty);
}
.jp-CaretDownEmptyThinIcon {
  background-image: var(--jp-icon-caret-down-empty-thin);
}
.jp-CaretDownIcon {
  background-image: var(--jp-icon-caret-down);
}
.jp-CaretLeftIcon {
  background-image: var(--jp-icon-caret-left);
}
.jp-CaretRightIcon {
  background-image: var(--jp-icon-caret-right);
}
.jp-CaretUpEmptyThinIcon {
  background-image: var(--jp-icon-caret-up-empty-thin);
}
.jp-CaretUpIcon {
  background-image: var(--jp-icon-caret-up);
}
.jp-CaseSensitiveIcon {
  background-image: var(--jp-icon-case-sensitive);
}
.jp-CheckIcon {
  background-image: var(--jp-icon-check);
}
.jp-CircleEmptyIcon {
  background-image: var(--jp-icon-circle-empty);
}
.jp-CircleIcon {
  background-image: var(--jp-icon-circle);
}
.jp-ClearIcon {
  background-image: var(--jp-icon-clear);
}
.jp-CloseIcon {
  background-image: var(--jp-icon-close);
}
.jp-CodeIcon {
  background-image: var(--jp-icon-code);
}
.jp-ConsoleIcon {
  background-image: var(--jp-icon-console);
}
.jp-CopyIcon {
  background-image: var(--jp-icon-copy);
}
.jp-CutIcon {
  background-image: var(--jp-icon-cut);
}
.jp-DownloadIcon {
  background-image: var(--jp-icon-download);
}
.jp-EditIcon {
  background-image: var(--jp-icon-edit);
}
.jp-EllipsesIcon {
  background-image: var(--jp-icon-ellipses);
}
.jp-ExtensionIcon {
  background-image: var(--jp-icon-extension);
}
.jp-FastForwardIcon {
  background-image: var(--jp-icon-fast-forward);
}
.jp-FileIcon {
  background-image: var(--jp-icon-file);
}
.jp-FileUploadIcon {
  background-image: var(--jp-icon-file-upload);
}
.jp-FilterListIcon {
  background-image: var(--jp-icon-filter-list);
}
.jp-FolderIcon {
  background-image: var(--jp-icon-folder);
}
.jp-Html5Icon {
  background-image: var(--jp-icon-html5);
}
.jp-ImageIcon {
  background-image: var(--jp-icon-image);
}
.jp-InspectorIcon {
  background-image: var(--jp-icon-inspector);
}
.jp-JsonIcon {
  background-image: var(--jp-icon-json);
}
.jp-JupyterFaviconIcon {
  background-image: var(--jp-icon-jupyter-favicon);
}
.jp-JupyterIcon {
  background-image: var(--jp-icon-jupyter);
}
.jp-JupyterlabWordmarkIcon {
  background-image: var(--jp-icon-jupyterlab-wordmark);
}
.jp-KernelIcon {
  background-image: var(--jp-icon-kernel);
}
.jp-KeyboardIcon {
  background-image: var(--jp-icon-keyboard);
}
.jp-LauncherIcon {
  background-image: var(--jp-icon-launcher);
}
.jp-LineFormIcon {
  background-image: var(--jp-icon-line-form);
}
.jp-LinkIcon {
  background-image: var(--jp-icon-link);
}
.jp-ListIcon {
  background-image: var(--jp-icon-list);
}
.jp-ListingsInfoIcon {
  background-image: var(--jp-icon-listings-info);
}
.jp-MarkdownIcon {
  background-image: var(--jp-icon-markdown);
}
.jp-NewFolderIcon {
  background-image: var(--jp-icon-new-folder);
}
.jp-NotTrustedIcon {
  background-image: var(--jp-icon-not-trusted);
}
.jp-NotebookIcon {
  background-image: var(--jp-icon-notebook);
}
.jp-NumberingIcon {
  background-image: var(--jp-icon-numbering);
}
.jp-OfflineBoltIcon {
  background-image: var(--jp-icon-offline-bolt);
}
.jp-PaletteIcon {
  background-image: var(--jp-icon-palette);
}
.jp-PasteIcon {
  background-image: var(--jp-icon-paste);
}
.jp-PdfIcon {
  background-image: var(--jp-icon-pdf);
}
.jp-PythonIcon {
  background-image: var(--jp-icon-python);
}
.jp-RKernelIcon {
  background-image: var(--jp-icon-r-kernel);
}
.jp-ReactIcon {
  background-image: var(--jp-icon-react);
}
.jp-RedoIcon {
  background-image: var(--jp-icon-redo);
}
.jp-RefreshIcon {
  background-image: var(--jp-icon-refresh);
}
.jp-RegexIcon {
  background-image: var(--jp-icon-regex);
}
.jp-RunIcon {
  background-image: var(--jp-icon-run);
}
.jp-RunningIcon {
  background-image: var(--jp-icon-running);
}
.jp-SaveIcon {
  background-image: var(--jp-icon-save);
}
.jp-SearchIcon {
  background-image: var(--jp-icon-search);
}
.jp-SettingsIcon {
  background-image: var(--jp-icon-settings);
}
.jp-SpreadsheetIcon {
  background-image: var(--jp-icon-spreadsheet);
}
.jp-StopIcon {
  background-image: var(--jp-icon-stop);
}
.jp-TabIcon {
  background-image: var(--jp-icon-tab);
}
.jp-TableRowsIcon {
  background-image: var(--jp-icon-table-rows);
}
.jp-TagIcon {
  background-image: var(--jp-icon-tag);
}
.jp-TerminalIcon {
  background-image: var(--jp-icon-terminal);
}
.jp-TextEditorIcon {
  background-image: var(--jp-icon-text-editor);
}
.jp-TocIcon {
  background-image: var(--jp-icon-toc);
}
.jp-TreeViewIcon {
  background-image: var(--jp-icon-tree-view);
}
.jp-TrustedIcon {
  background-image: var(--jp-icon-trusted);
}
.jp-UndoIcon {
  background-image: var(--jp-icon-undo);
}
.jp-VegaIcon {
  background-image: var(--jp-icon-vega);
}
.jp-YamlIcon {
  background-image: var(--jp-icon-yaml);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

:root {
  --jp-icon-search-white: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
}

.jp-Icon,
.jp-MaterialIcon {
  background-position: center;
  background-repeat: no-repeat;
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-cover {
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}

/**
 * (DEPRECATED) Support for specific CSS icon sizes
 */

.jp-Icon-16 {
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-18 {
  background-size: 18px;
  min-width: 18px;
  min-height: 18px;
}

.jp-Icon-20 {
  background-size: 20px;
  min-width: 20px;
  min-height: 20px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for icons as inline SVG HTMLElements
 */

/* recolor the primary elements of an icon */
.jp-icon0[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon1[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon2[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}
/* recolor the accent elements of an icon */
.jp-icon-accent0[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-accent1[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-accent2[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-accent3[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-accent4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-accent0[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-accent1[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-accent2[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-accent3[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-accent4[stroke] {
  stroke: var(--jp-layout-color4);
}
/* set the color of an icon to transparent */
.jp-icon-none[fill] {
  fill: none;
}

.jp-icon-none[stroke] {
  stroke: none;
}
/* brand icon colors. Same for light and dark */
.jp-icon-brand0[fill] {
  fill: var(--jp-brand-color0);
}
.jp-icon-brand1[fill] {
  fill: var(--jp-brand-color1);
}
.jp-icon-brand2[fill] {
  fill: var(--jp-brand-color2);
}
.jp-icon-brand3[fill] {
  fill: var(--jp-brand-color3);
}
.jp-icon-brand4[fill] {
  fill: var(--jp-brand-color4);
}

.jp-icon-brand0[stroke] {
  stroke: var(--jp-brand-color0);
}
.jp-icon-brand1[stroke] {
  stroke: var(--jp-brand-color1);
}
.jp-icon-brand2[stroke] {
  stroke: var(--jp-brand-color2);
}
.jp-icon-brand3[stroke] {
  stroke: var(--jp-brand-color3);
}
.jp-icon-brand4[stroke] {
  stroke: var(--jp-brand-color4);
}
/* warn icon colors. Same for light and dark */
.jp-icon-warn0[fill] {
  fill: var(--jp-warn-color0);
}
.jp-icon-warn1[fill] {
  fill: var(--jp-warn-color1);
}
.jp-icon-warn2[fill] {
  fill: var(--jp-warn-color2);
}
.jp-icon-warn3[fill] {
  fill: var(--jp-warn-color3);
}

.jp-icon-warn0[stroke] {
  stroke: var(--jp-warn-color0);
}
.jp-icon-warn1[stroke] {
  stroke: var(--jp-warn-color1);
}
.jp-icon-warn2[stroke] {
  stroke: var(--jp-warn-color2);
}
.jp-icon-warn3[stroke] {
  stroke: var(--jp-warn-color3);
}
/* icon colors that contrast well with each other and most backgrounds */
.jp-icon-contrast0[fill] {
  fill: var(--jp-icon-contrast-color0);
}
.jp-icon-contrast1[fill] {
  fill: var(--jp-icon-contrast-color1);
}
.jp-icon-contrast2[fill] {
  fill: var(--jp-icon-contrast-color2);
}
.jp-icon-contrast3[fill] {
  fill: var(--jp-icon-contrast-color3);
}

.jp-icon-contrast0[stroke] {
  stroke: var(--jp-icon-contrast-color0);
}
.jp-icon-contrast1[stroke] {
  stroke: var(--jp-icon-contrast-color1);
}
.jp-icon-contrast2[stroke] {
  stroke: var(--jp-icon-contrast-color2);
}
.jp-icon-contrast3[stroke] {
  stroke: var(--jp-icon-contrast-color3);
}

/* CSS for icons in selected items in the settings editor */
#setting-editor .jp-PluginList .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}
#setting-editor
  .jp-PluginList
  .jp-mod-selected
  .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* CSS for icons in selected filebrowser listing items */
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* CSS for icons in selected tabs in the sidebar tab manager */
#tab-manager .lm-TabBar-tab.jp-mod-active .jp-icon-selectable[fill] {
  fill: #fff;
}

#tab-manager .lm-TabBar-tab.jp-mod-active .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}
#tab-manager
  .lm-TabBar-tab.jp-mod-active
  .jp-icon-hover
  :hover
  .jp-icon-selectable[fill] {
  fill: var(--jp-brand-color1);
}

#tab-manager
  .lm-TabBar-tab.jp-mod-active
  .jp-icon-hover
  :hover
  .jp-icon-selectable-inverse[fill] {
  fill: #fff;
}

/**
 * TODO: come up with non css-hack solution for showing the busy icon on top
 *  of the close icon
 * CSS for complex behavior of close icon of tabs in the sidebar tab manager
 */
#tab-manager
  .lm-TabBar-tab.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}
#tab-manager
  .lm-TabBar-tab.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

#tab-manager
  .lm-TabBar-tab.jp-mod-dirty.jp-mod-active
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: #fff;
}

/**
* TODO: come up with non css-hack solution for showing the busy icon on top
*  of the close icon
* CSS for complex behavior of close icon of tabs in the main area tabbar
*/
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

/* CSS for icons in status bar */
#jp-main-statusbar .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

#jp-main-statusbar .jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}
/* special handling for splash icon CSS. While the theme CSS reloads during
   splash, the splash icon can loose theming. To prevent that, we set a
   default for its color variable */
:root {
  --jp-warn-color0: var(--md-orange-700);
}

/* not sure what to do with this one, used in filebrowser listing */
.jp-DragIcon {
  margin-right: 4px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for alt colors for icons as inline SVG HTMLElements
 */

/* alt recolor the primary elements of an icon */
.jp-icon-alt .jp-icon0[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-alt .jp-icon1[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-alt .jp-icon2[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-alt .jp-icon3[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-alt .jp-icon4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-alt .jp-icon0[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-alt .jp-icon1[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-alt .jp-icon2[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-alt .jp-icon3[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-alt .jp-icon4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* alt recolor the accent elements of an icon */
.jp-icon-alt .jp-icon-accent0[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-alt .jp-icon-accent1[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-alt .jp-icon-accent2[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-alt .jp-icon-accent3[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-alt .jp-icon-accent4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-alt .jp-icon-accent0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-alt .jp-icon-accent1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-alt .jp-icon-accent2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-alt .jp-icon-accent3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-alt .jp-icon-accent4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-icon-hoverShow:not(:hover) svg {
  display: none !important;
}

/**
 * Support for hover colors for icons as inline SVG HTMLElements
 */

/**
 * regular colors
 */

/* recolor the primary elements of an icon */
.jp-icon-hover :hover .jp-icon0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-hover :hover .jp-icon1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-hover :hover .jp-icon2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-hover :hover .jp-icon3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-hover :hover .jp-icon4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-hover :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-hover :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-hover :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-hover :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-hover :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-hover :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-hover :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-hover :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-hover :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-hover :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-hover :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-hover :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-hover :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-hover :hover .jp-icon-none-hover[fill] {
  fill: none;
}

.jp-icon-hover :hover .jp-icon-none-hover[stroke] {
  stroke: none;
}

/**
 * inverse colors
 */

/* inverse recolor the primary elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* inverse recolor the accent elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-switch {
  display: flex;
  align-items: center;
  padding-left: 4px;
  padding-right: 4px;
  font-size: var(--jp-ui-font-size1);
  background-color: transparent;
  color: var(--jp-ui-font-color1);
  border: none;
  height: 20px;
}

.jp-switch:hover {
  background-color: var(--jp-layout-color2);
}

.jp-switch-label {
  margin-right: 5px;
}

.jp-switch-track {
  cursor: pointer;
  background-color: var(--jp-border-color1);
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 34px;
  height: 16px;
  width: 35px;
  position: relative;
}

.jp-switch-track::before {
  content: '';
  position: absolute;
  height: 10px;
  width: 10px;
  margin: 3px;
  left: 0px;
  background-color: var(--jp-ui-inverse-font-color1);
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 50%;
}

.jp-switch[aria-checked='true'] .jp-switch-track {
  background-color: var(--jp-warn-color0);
}

.jp-switch[aria-checked='true'] .jp-switch-track::before {
  /* track width (35) - margins (3 + 3) - thumb width (10) */
  left: 19px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Sibling imports */

/* Override Blueprint's _reset.scss styles */
html {
  box-sizing: unset;
}

*,
*::before,
*::after {
  box-sizing: unset;
}

body {
  color: unset;
  font-family: var(--jp-ui-font-family);
}

p {
  margin-top: unset;
  margin-bottom: unset;
}

small {
  font-size: unset;
}

strong {
  font-weight: unset;
}

/* Override Blueprint's _typography.scss styles */
a {
  text-decoration: unset;
  color: unset;
}
a:hover {
  text-decoration: unset;
  color: unset;
}

/* Override Blueprint's _accessibility.scss styles */
:focus {
  outline: unset;
  outline-offset: unset;
  -moz-outline-radius: unset;
}

/* Styles for ui-components */
.jp-Button {
  border-radius: var(--jp-border-radius);
  padding: 0px 12px;
  font-size: var(--jp-ui-font-size1);
}

/* Use our own theme for hover styles */
button.jp-Button.bp3-button.bp3-minimal:hover {
  background-color: var(--jp-layout-color2);
}
.jp-Button.minimal {
  color: unset !important;
}

.jp-Button.jp-ToolbarButtonComponent {
  text-transform: none;
}

.jp-InputGroup input {
  box-sizing: border-box;
  border-radius: 0;
  background-color: transparent;
  color: var(--jp-ui-font-color0);
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.jp-InputGroup input:focus {
  box-shadow: inset 0 0 0 var(--jp-border-width)
      var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-InputGroup input::placeholder,
input::placeholder {
  color: var(--jp-ui-font-color3);
}

.jp-BPIcon {
  display: inline-block;
  vertical-align: middle;
  margin: auto;
}

/* Stop blueprint futzing with our icon fills */
.bp3-icon.jp-BPIcon > svg:not([fill]) {
  fill: var(--jp-inverse-layout-color3);
}

.jp-InputGroupAction {
  padding: 6px;
}

.jp-HTMLSelect.jp-DefaultStyle select {
  background-color: initial;
  border: none;
  border-radius: 0;
  box-shadow: none;
  color: var(--jp-ui-font-color0);
  display: block;
  font-size: var(--jp-ui-font-size1);
  height: 24px;
  line-height: 14px;
  padding: 0 25px 0 10px;
  text-align: left;
  -moz-appearance: none;
  -webkit-appearance: none;
}

/* Use our own theme for hover and option styles */
.jp-HTMLSelect.jp-DefaultStyle select:hover,
.jp-HTMLSelect.jp-DefaultStyle select > option {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color0);
}
select {
  box-sizing: border-box;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapse {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-top: 1px solid var(--jp-border-color2);
  border-bottom: 1px solid var(--jp-border-color2);
}

.jp-Collapse-header {
  padding: 1px 12px;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color1);
  font-size: var(--jp-ui-font-size2);
}

.jp-Collapse-header:hover {
  background-color: var(--jp-layout-color2);
}

.jp-Collapse-contents {
  padding: 0px 12px 0px 12px;
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-commandpalette-search-height: 28px;
}

/*-----------------------------------------------------------------------------
| Overall styles
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  padding-bottom: 0px;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Modal variant
|----------------------------------------------------------------------------*/

.jp-ModalCommandPalette {
  position: absolute;
  z-index: 10000;
  top: 38px;
  left: 30%;
  margin: 0;
  padding: 4px;
  width: 40%;
  box-shadow: var(--jp-elevation-z4);
  border-radius: 4px;
  background: var(--jp-layout-color0);
}

.jp-ModalCommandPalette .lm-CommandPalette {
  max-height: 40vh;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-close-icon::after {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-header {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-item {
  margin-left: 4px;
  margin-right: 4px;
}

.jp-ModalCommandPalette
  .lm-CommandPalette
  .lm-CommandPalette-item.lm-mod-disabled {
  display: none;
}

/*-----------------------------------------------------------------------------
| Search
|----------------------------------------------------------------------------*/

.lm-CommandPalette-search {
  padding: 4px;
  background-color: var(--jp-layout-color1);
  z-index: 2;
}

.lm-CommandPalette-wrapper {
  overflow: overlay;
  padding: 0px 9px;
  background-color: var(--jp-input-active-background);
  height: 30px;
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.lm-CommandPalette.lm-mod-focused .lm-CommandPalette-wrapper {
  box-shadow: inset 0 0 0 1px var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.lm-CommandPalette-wrapper::after {
  content: ' ';
  color: white;
  background-color: var(--jp-brand-color1);
  position: absolute;
  top: 4px;
  right: 4px;
  height: 30px;
  width: 10px;
  padding: 0px 10px;
  background-image: var(--jp-icon-search-white);
  background-size: 20px;
  background-repeat: no-repeat;
  background-position: center;
}

.lm-CommandPalette-input {
  background: transparent;
  width: calc(100% - 18px);
  float: left;
  border: none;
  outline: none;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  line-height: var(--jp-private-commandpalette-search-height);
}

.lm-CommandPalette-input::-webkit-input-placeholder,
.lm-CommandPalette-input::-moz-placeholder,
.lm-CommandPalette-input:-ms-input-placeholder {
  color: var(--jp-ui-font-color3);
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Results
|----------------------------------------------------------------------------*/

.lm-CommandPalette-header:first-child {
  margin-top: 0px;
}

.lm-CommandPalette-header {
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin-top: 8px;
  padding: 8px 0 8px 12px;
  text-transform: uppercase;
}

.lm-CommandPalette-header.lm-mod-active {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-header > mark {
  background-color: transparent;
  font-weight: bold;
  color: var(--jp-ui-font-color1);
}

.lm-CommandPalette-item {
  padding: 4px 12px 4px 4px;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  font-weight: 400;
  display: flex;
}

.lm-CommandPalette-item.lm-mod-disabled {
  color: var(--jp-ui-font-color3);
}

.lm-CommandPalette-item.lm-mod-active {
  background: var(--jp-layout-color3);
}

.lm-CommandPalette-item.lm-mod-active:hover:not(.lm-mod-disabled) {
  background: var(--jp-layout-color4);
}

.lm-CommandPalette-item:hover:not(.lm-mod-active):not(.lm-mod-disabled) {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-itemContent {
  overflow: hidden;
}

.lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.lm-CommandPalette-item.lm-mod-disabled mark {
  color: var(--jp-ui-font-color3);
}

.lm-CommandPalette-item .lm-CommandPalette-itemIcon {
  margin: 0 4px 0 0;
  position: relative;
  width: 16px;
  top: 2px;
  flex: 0 0 auto;
}

.lm-CommandPalette-item.lm-mod-disabled .lm-CommandPalette-itemIcon {
  opacity: 0.4;
}

.lm-CommandPalette-item .lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemCaption {
  display: none;
}

.lm-CommandPalette-content {
  background-color: var(--jp-layout-color1);
}

.lm-CommandPalette-content:empty:after {
  content: 'No results';
  margin: auto;
  margin-top: 20px;
  width: 100px;
  display: block;
  font-size: var(--jp-ui-font-size2);
  font-family: var(--jp-ui-font-family);
  font-weight: lighter;
}

.lm-CommandPalette-emptyMessage {
  text-align: center;
  margin-top: 24px;
  line-height: 1.32;
  padding: 0px 8px;
  color: var(--jp-content-font-color3);
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Dialog {
  position: absolute;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  top: 0px;
  left: 0px;
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-dialog-background);
}

.jp-Dialog-content {
  display: flex;
  flex-direction: column;
  margin-left: auto;
  margin-right: auto;
  background: var(--jp-layout-color1);
  padding: 24px;
  padding-bottom: 12px;
  min-width: 300px;
  min-height: 150px;
  max-width: 1000px;
  max-height: 500px;
  box-sizing: border-box;
  box-shadow: var(--jp-elevation-z20);
  word-wrap: break-word;
  border-radius: var(--jp-border-radius);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color1);
  resize: both;
}

.jp-Dialog-button {
  overflow: visible;
}

button.jp-Dialog-button:focus {
  outline: 1px solid var(--jp-brand-color1);
  outline-offset: 4px;
  -moz-outline-radius: 0px;
}

button.jp-Dialog-button:focus::-moz-focus-inner {
  border: 0;
}

button.jp-Dialog-close-button {
  padding: 0;
  height: 100%;
  min-width: unset;
  min-height: unset;
}

.jp-Dialog-header {
  display: flex;
  justify-content: space-between;
  flex: 0 0 auto;
  padding-bottom: 12px;
  font-size: var(--jp-ui-font-size3);
  font-weight: 400;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-body {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  font-size: var(--jp-ui-font-size1);
  background: var(--jp-layout-color1);
  overflow: auto;
}

.jp-Dialog-footer {
  display: flex;
  flex-direction: row;
  justify-content: flex-end;
  flex: 0 0 auto;
  margin-left: -12px;
  margin-right: -12px;
  padding: 12px;
}

.jp-Dialog-title {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.jp-Dialog-body > .jp-select-wrapper {
  width: 100%;
}

.jp-Dialog-body > button {
  padding: 0px 16px;
}

.jp-Dialog-body > label {
  line-height: 1.4;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-button.jp-mod-styled:not(:last-child) {
  margin-right: 12px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-HoverBox {
  position: fixed;
}

.jp-HoverBox.jp-mod-outofview {
  display: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-IFrame {
  width: 100%;
  height: 100%;
}

.jp-IFrame > iframe {
  border: none;
}

/*
When drag events occur, `p-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-IFrame {
  position: relative;
}

body.lm-mod-override-cursor .jp-IFrame:before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MainAreaWidget > :focus {
  outline: none;
}

/**
 * google-material-color v1.2.6
 * https://github.com/danlevan/google-material-color
 */
:root {
  --md-red-50: #ffebee;
  --md-red-100: #ffcdd2;
  --md-red-200: #ef9a9a;
  --md-red-300: #e57373;
  --md-red-400: #ef5350;
  --md-red-500: #f44336;
  --md-red-600: #e53935;
  --md-red-700: #d32f2f;
  --md-red-800: #c62828;
  --md-red-900: #b71c1c;
  --md-red-A100: #ff8a80;
  --md-red-A200: #ff5252;
  --md-red-A400: #ff1744;
  --md-red-A700: #d50000;

  --md-pink-50: #fce4ec;
  --md-pink-100: #f8bbd0;
  --md-pink-200: #f48fb1;
  --md-pink-300: #f06292;
  --md-pink-400: #ec407a;
  --md-pink-500: #e91e63;
  --md-pink-600: #d81b60;
  --md-pink-700: #c2185b;
  --md-pink-800: #ad1457;
  --md-pink-900: #880e4f;
  --md-pink-A100: #ff80ab;
  --md-pink-A200: #ff4081;
  --md-pink-A400: #f50057;
  --md-pink-A700: #c51162;

  --md-purple-50: #f3e5f5;
  --md-purple-100: #e1bee7;
  --md-purple-200: #ce93d8;
  --md-purple-300: #ba68c8;
  --md-purple-400: #ab47bc;
  --md-purple-500: #9c27b0;
  --md-purple-600: #8e24aa;
  --md-purple-700: #7b1fa2;
  --md-purple-800: #6a1b9a;
  --md-purple-900: #4a148c;
  --md-purple-A100: #ea80fc;
  --md-purple-A200: #e040fb;
  --md-purple-A400: #d500f9;
  --md-purple-A700: #aa00ff;

  --md-deep-purple-50: #ede7f6;
  --md-deep-purple-100: #d1c4e9;
  --md-deep-purple-200: #b39ddb;
  --md-deep-purple-300: #9575cd;
  --md-deep-purple-400: #7e57c2;
  --md-deep-purple-500: #673ab7;
  --md-deep-purple-600: #5e35b1;
  --md-deep-purple-700: #512da8;
  --md-deep-purple-800: #4527a0;
  --md-deep-purple-900: #311b92;
  --md-deep-purple-A100: #b388ff;
  --md-deep-purple-A200: #7c4dff;
  --md-deep-purple-A400: #651fff;
  --md-deep-purple-A700: #6200ea;

  --md-indigo-50: #e8eaf6;
  --md-indigo-100: #c5cae9;
  --md-indigo-200: #9fa8da;
  --md-indigo-300: #7986cb;
  --md-indigo-400: #5c6bc0;
  --md-indigo-500: #3f51b5;
  --md-indigo-600: #3949ab;
  --md-indigo-700: #303f9f;
  --md-indigo-800: #283593;
  --md-indigo-900: #1a237e;
  --md-indigo-A100: #8c9eff;
  --md-indigo-A200: #536dfe;
  --md-indigo-A400: #3d5afe;
  --md-indigo-A700: #304ffe;

  --md-blue-50: #e3f2fd;
  --md-blue-100: #bbdefb;
  --md-blue-200: #90caf9;
  --md-blue-300: #64b5f6;
  --md-blue-400: #42a5f5;
  --md-blue-500: #2196f3;
  --md-blue-600: #1e88e5;
  --md-blue-700: #1976d2;
  --md-blue-800: #1565c0;
  --md-blue-900: #0d47a1;
  --md-blue-A100: #82b1ff;
  --md-blue-A200: #448aff;
  --md-blue-A400: #2979ff;
  --md-blue-A700: #2962ff;

  --md-light-blue-50: #e1f5fe;
  --md-light-blue-100: #b3e5fc;
  --md-light-blue-200: #81d4fa;
  --md-light-blue-300: #4fc3f7;
  --md-light-blue-400: #29b6f6;
  --md-light-blue-500: #03a9f4;
  --md-light-blue-600: #039be5;
  --md-light-blue-700: #0288d1;
  --md-light-blue-800: #0277bd;
  --md-light-blue-900: #01579b;
  --md-light-blue-A100: #80d8ff;
  --md-light-blue-A200: #40c4ff;
  --md-light-blue-A400: #00b0ff;
  --md-light-blue-A700: #0091ea;

  --md-cyan-50: #e0f7fa;
  --md-cyan-100: #b2ebf2;
  --md-cyan-200: #80deea;
  --md-cyan-300: #4dd0e1;
  --md-cyan-400: #26c6da;
  --md-cyan-500: #00bcd4;
  --md-cyan-600: #00acc1;
  --md-cyan-700: #0097a7;
  --md-cyan-800: #00838f;
  --md-cyan-900: #006064;
  --md-cyan-A100: #84ffff;
  --md-cyan-A200: #18ffff;
  --md-cyan-A400: #00e5ff;
  --md-cyan-A700: #00b8d4;

  --md-teal-50: #e0f2f1;
  --md-teal-100: #b2dfdb;
  --md-teal-200: #80cbc4;
  --md-teal-300: #4db6ac;
  --md-teal-400: #26a69a;
  --md-teal-500: #009688;
  --md-teal-600: #00897b;
  --md-teal-700: #00796b;
  --md-teal-800: #00695c;
  --md-teal-900: #004d40;
  --md-teal-A100: #a7ffeb;
  --md-teal-A200: #64ffda;
  --md-teal-A400: #1de9b6;
  --md-teal-A700: #00bfa5;

  --md-green-50: #e8f5e9;
  --md-green-100: #c8e6c9;
  --md-green-200: #a5d6a7;
  --md-green-300: #81c784;
  --md-green-400: #66bb6a;
  --md-green-500: #4caf50;
  --md-green-600: #43a047;
  --md-green-700: #388e3c;
  --md-green-800: #2e7d32;
  --md-green-900: #1b5e20;
  --md-green-A100: #b9f6ca;
  --md-green-A200: #69f0ae;
  --md-green-A400: #00e676;
  --md-green-A700: #00c853;

  --md-light-green-50: #f1f8e9;
  --md-light-green-100: #dcedc8;
  --md-light-green-200: #c5e1a5;
  --md-light-green-300: #aed581;
  --md-light-green-400: #9ccc65;
  --md-light-green-500: #8bc34a;
  --md-light-green-600: #7cb342;
  --md-light-green-700: #689f38;
  --md-light-green-800: #558b2f;
  --md-light-green-900: #33691e;
  --md-light-green-A100: #ccff90;
  --md-light-green-A200: #b2ff59;
  --md-light-green-A400: #76ff03;
  --md-light-green-A700: #64dd17;

  --md-lime-50: #f9fbe7;
  --md-lime-100: #f0f4c3;
  --md-lime-200: #e6ee9c;
  --md-lime-300: #dce775;
  --md-lime-400: #d4e157;
  --md-lime-500: #cddc39;
  --md-lime-600: #c0ca33;
  --md-lime-700: #afb42b;
  --md-lime-800: #9e9d24;
  --md-lime-900: #827717;
  --md-lime-A100: #f4ff81;
  --md-lime-A200: #eeff41;
  --md-lime-A400: #c6ff00;
  --md-lime-A700: #aeea00;

  --md-yellow-50: #fffde7;
  --md-yellow-100: #fff9c4;
  --md-yellow-200: #fff59d;
  --md-yellow-300: #fff176;
  --md-yellow-400: #ffee58;
  --md-yellow-500: #ffeb3b;
  --md-yellow-600: #fdd835;
  --md-yellow-700: #fbc02d;
  --md-yellow-800: #f9a825;
  --md-yellow-900: #f57f17;
  --md-yellow-A100: #ffff8d;
  --md-yellow-A200: #ffff00;
  --md-yellow-A400: #ffea00;
  --md-yellow-A700: #ffd600;

  --md-amber-50: #fff8e1;
  --md-amber-100: #ffecb3;
  --md-amber-200: #ffe082;
  --md-amber-300: #ffd54f;
  --md-amber-400: #ffca28;
  --md-amber-500: #ffc107;
  --md-amber-600: #ffb300;
  --md-amber-700: #ffa000;
  --md-amber-800: #ff8f00;
  --md-amber-900: #ff6f00;
  --md-amber-A100: #ffe57f;
  --md-amber-A200: #ffd740;
  --md-amber-A400: #ffc400;
  --md-amber-A700: #ffab00;

  --md-orange-50: #fff3e0;
  --md-orange-100: #ffe0b2;
  --md-orange-200: #ffcc80;
  --md-orange-300: #ffb74d;
  --md-orange-400: #ffa726;
  --md-orange-500: #ff9800;
  --md-orange-600: #fb8c00;
  --md-orange-700: #f57c00;
  --md-orange-800: #ef6c00;
  --md-orange-900: #e65100;
  --md-orange-A100: #ffd180;
  --md-orange-A200: #ffab40;
  --md-orange-A400: #ff9100;
  --md-orange-A700: #ff6d00;

  --md-deep-orange-50: #fbe9e7;
  --md-deep-orange-100: #ffccbc;
  --md-deep-orange-200: #ffab91;
  --md-deep-orange-300: #ff8a65;
  --md-deep-orange-400: #ff7043;
  --md-deep-orange-500: #ff5722;
  --md-deep-orange-600: #f4511e;
  --md-deep-orange-700: #e64a19;
  --md-deep-orange-800: #d84315;
  --md-deep-orange-900: #bf360c;
  --md-deep-orange-A100: #ff9e80;
  --md-deep-orange-A200: #ff6e40;
  --md-deep-orange-A400: #ff3d00;
  --md-deep-orange-A700: #dd2c00;

  --md-brown-50: #efebe9;
  --md-brown-100: #d7ccc8;
  --md-brown-200: #bcaaa4;
  --md-brown-300: #a1887f;
  --md-brown-400: #8d6e63;
  --md-brown-500: #795548;
  --md-brown-600: #6d4c41;
  --md-brown-700: #5d4037;
  --md-brown-800: #4e342e;
  --md-brown-900: #3e2723;

  --md-grey-50: #fafafa;
  --md-grey-100: #f5f5f5;
  --md-grey-200: #eeeeee;
  --md-grey-300: #e0e0e0;
  --md-grey-400: #bdbdbd;
  --md-grey-500: #9e9e9e;
  --md-grey-600: #757575;
  --md-grey-700: #616161;
  --md-grey-800: #424242;
  --md-grey-900: #212121;

  --md-blue-grey-50: #eceff1;
  --md-blue-grey-100: #cfd8dc;
  --md-blue-grey-200: #b0bec5;
  --md-blue-grey-300: #90a4ae;
  --md-blue-grey-400: #78909c;
  --md-blue-grey-500: #607d8b;
  --md-blue-grey-600: #546e7a;
  --md-blue-grey-700: #455a64;
  --md-blue-grey-800: #37474f;
  --md-blue-grey-900: #263238;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Spinner {
  position: absolute;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-layout-color0);
  outline: none;
}

.jp-SpinnerContent {
  font-size: 10px;
  margin: 50px auto;
  text-indent: -9999em;
  width: 3em;
  height: 3em;
  border-radius: 50%;
  background: var(--jp-brand-color3);
  background: linear-gradient(
    to right,
    #f37626 10%,
    rgba(255, 255, 255, 0) 42%
  );
  position: relative;
  animation: load3 1s infinite linear, fadeIn 1s;
}

.jp-SpinnerContent:before {
  width: 50%;
  height: 50%;
  background: #f37626;
  border-radius: 100% 0 0 0;
  position: absolute;
  top: 0;
  left: 0;
  content: '';
}

.jp-SpinnerContent:after {
  background: var(--jp-layout-color0);
  width: 75%;
  height: 75%;
  border-radius: 50%;
  content: '';
  margin: auto;
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
}

@keyframes fadeIn {
  0% {
    opacity: 0;
  }
  100% {
    opacity: 1;
  }
}

@keyframes load3 {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

button.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: none;
  box-sizing: border-box;
  text-align: center;
  line-height: 32px;
  height: 32px;
  padding: 0px 12px;
  letter-spacing: 0.8px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled {
  background: var(--jp-input-background);
  height: 28px;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color1);
  padding-left: 7px;
  padding-right: 7px;
  font-size: var(--jp-ui-font-size2);
  color: var(--jp-ui-font-color0);
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled:focus {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-select-wrapper {
  display: flex;
  position: relative;
  flex-direction: column;
  padding: 1px;
  background-color: var(--jp-layout-color1);
  height: 28px;
  box-sizing: border-box;
  margin-bottom: 12px;
}

.jp-select-wrapper.jp-mod-focused select.jp-mod-styled {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-input-active-background);
}

select.jp-mod-styled:hover {
  background-color: var(--jp-layout-color1);
  cursor: pointer;
  color: var(--jp-ui-font-color0);
  background-color: var(--jp-input-hover-background);
  box-shadow: inset 0 0px 1px rgba(0, 0, 0, 0.5);
}

select.jp-mod-styled {
  flex: 1 1 auto;
  height: 32px;
  width: 100%;
  font-size: var(--jp-ui-font-size2);
  background: var(--jp-input-background);
  color: var(--jp-ui-font-color0);
  padding: 0 25px 0 8px;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toolbar-height: calc(
    28px + var(--jp-border-width)
  ); /* leave 28px for content */
}

.jp-Toolbar {
  color: var(--jp-ui-font-color1);
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: 2px;
  z-index: 1;
  overflow-x: hidden;
}

.jp-Toolbar:hover {
  overflow-x: auto;
}

/* Toolbar items */

.jp-Toolbar > .jp-Toolbar-item.jp-Toolbar-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-Toolbar-item.jp-Toolbar-kernelStatus {
  display: inline-block;
  width: 32px;
  background-repeat: no-repeat;
  background-position: center;
  background-size: 16px;
}

.jp-Toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  display: flex;
  padding-left: 1px;
  padding-right: 1px;
  font-size: var(--jp-ui-font-size1);
  line-height: var(--jp-private-toolbar-height);
  height: 100%;
}

/* Toolbar buttons */

/* This is the div we use to wrap the react component into a Widget */
div.jp-ToolbarButton {
  color: transparent;
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0px;
  margin: 0px;
}

button.jp-ToolbarButtonComponent {
  background: var(--jp-layout-color1);
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0px 6px;
  margin: 0px;
  height: 24px;
  border-radius: var(--jp-border-radius);
  display: flex;
  align-items: center;
  text-align: center;
  font-size: 14px;
  min-width: unset;
  min-height: unset;
}

button.jp-ToolbarButtonComponent:disabled {
  opacity: 0.4;
}

button.jp-ToolbarButtonComponent span {
  padding: 0px;
  flex: 0 0 auto;
}

button.jp-ToolbarButtonComponent .jp-ToolbarButtonComponent-label {
  font-size: var(--jp-ui-font-size1);
  line-height: 100%;
  padding-left: 2px;
  color: var(--jp-ui-font-color1);
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar.jp-Toolbar-micro {
  padding: 0;
  min-height: 0;
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar {
  border: none;
  box-shadow: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ body.p-mod-override-cursor *, /* </DEPRECATED> */
body.lm-mod-override-cursor * {
  cursor: inherit !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-JSONEditor {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.jp-JSONEditor-host {
  flex: 1 1 auto;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0px;
  background: var(--jp-layout-color0);
  min-height: 50px;
  padding: 1px;
}

.jp-JSONEditor.jp-mod-error .jp-JSONEditor-host {
  border-color: red;
  outline-color: red;
}

.jp-JSONEditor-header {
  display: flex;
  flex: 1 0 auto;
  padding: 0 0 0 12px;
}

.jp-JSONEditor-header label {
  flex: 0 0 auto;
}

.jp-JSONEditor-commitButton {
  height: 16px;
  width: 16px;
  background-size: 18px;
  background-repeat: no-repeat;
  background-position: center;
}

.jp-JSONEditor-host.jp-mod-focused {
  background-color: var(--jp-input-active-background);
  border: 1px solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

.jp-Editor.jp-mod-dropTarget {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

/* BASICS */

.CodeMirror {
  /* Set height, width, borders, and global font properties here */
  font-family: monospace;
  height: 300px;
  color: black;
  direction: ltr;
}

/* PADDING */

.CodeMirror-lines {
  padding: 4px 0; /* Vertical padding around content */
}
.CodeMirror pre.CodeMirror-line,
.CodeMirror pre.CodeMirror-line-like {
  padding: 0 4px; /* Horizontal padding of content */
}

.CodeMirror-scrollbar-filler, .CodeMirror-gutter-filler {
  background-color: white; /* The little square between H and V scrollbars */
}

/* GUTTER */

.CodeMirror-gutters {
  border-right: 1px solid #ddd;
  background-color: #f7f7f7;
  white-space: nowrap;
}
.CodeMirror-linenumbers {}
.CodeMirror-linenumber {
  padding: 0 3px 0 5px;
  min-width: 20px;
  text-align: right;
  color: #999;
  white-space: nowrap;
}

.CodeMirror-guttermarker { color: black; }
.CodeMirror-guttermarker-subtle { color: #999; }

/* CURSOR */

.CodeMirror-cursor {
  border-left: 1px solid black;
  border-right: none;
  width: 0;
}
/* Shown when moving in bi-directional text */
.CodeMirror div.CodeMirror-secondarycursor {
  border-left: 1px solid silver;
}
.cm-fat-cursor .CodeMirror-cursor {
  width: auto;
  border: 0 !important;
  background: #7e7;
}
.cm-fat-cursor div.CodeMirror-cursors {
  z-index: 1;
}
.cm-fat-cursor-mark {
  background-color: rgba(20, 255, 20, 0.5);
  -webkit-animation: blink 1.06s steps(1) infinite;
  -moz-animation: blink 1.06s steps(1) infinite;
  animation: blink 1.06s steps(1) infinite;
}
.cm-animate-fat-cursor {
  width: auto;
  border: 0;
  -webkit-animation: blink 1.06s steps(1) infinite;
  -moz-animation: blink 1.06s steps(1) infinite;
  animation: blink 1.06s steps(1) infinite;
  background-color: #7e7;
}
@-moz-keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}
@-webkit-keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}
@keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}

/* Can style cursor different in overwrite (non-insert) mode */
.CodeMirror-overwrite .CodeMirror-cursor {}

.cm-tab { display: inline-block; text-decoration: inherit; }

.CodeMirror-rulers {
  position: absolute;
  left: 0; right: 0; top: -50px; bottom: 0;
  overflow: hidden;
}
.CodeMirror-ruler {
  border-left: 1px solid #ccc;
  top: 0; bottom: 0;
  position: absolute;
}

/* DEFAULT THEME */

.cm-s-default .cm-header {color: blue;}
.cm-s-default .cm-quote {color: #090;}
.cm-negative {color: #d44;}
.cm-positive {color: #292;}
.cm-header, .cm-strong {font-weight: bold;}
.cm-em {font-style: italic;}
.cm-link {text-decoration: underline;}
.cm-strikethrough {text-decoration: line-through;}

.cm-s-default .cm-keyword {color: #708;}
.cm-s-default .cm-atom {color: #219;}
.cm-s-default .cm-number {color: #164;}
.cm-s-default .cm-def {color: #00f;}
.cm-s-default .cm-variable,
.cm-s-default .cm-punctuation,
.cm-s-default .cm-property,
.cm-s-default .cm-operator {}
.cm-s-default .cm-variable-2 {color: #05a;}
.cm-s-default .cm-variable-3, .cm-s-default .cm-type {color: #085;}
.cm-s-default .cm-comment {color: #a50;}
.cm-s-default .cm-string {color: #a11;}
.cm-s-default .cm-string-2 {color: #f50;}
.cm-s-default .cm-meta {color: #555;}
.cm-s-default .cm-qualifier {color: #555;}
.cm-s-default .cm-builtin {color: #30a;}
.cm-s-default .cm-bracket {color: #997;}
.cm-s-default .cm-tag {color: #170;}
.cm-s-default .cm-attribute {color: #00c;}
.cm-s-default .cm-hr {color: #999;}
.cm-s-default .cm-link {color: #00c;}

.cm-s-default .cm-error {color: #f00;}
.cm-invalidchar {color: #f00;}

.CodeMirror-composing { border-bottom: 2px solid; }

/* Default styles for common addons */

div.CodeMirror span.CodeMirror-matchingbracket {color: #0b0;}
div.CodeMirror span.CodeMirror-nonmatchingbracket {color: #a22;}
.CodeMirror-matchingtag { background: rgba(255, 150, 0, .3); }
.CodeMirror-activeline-background {background: #e8f2ff;}

/* STOP */

/* The rest of this file contains styles related to the mechanics of
   the editor. You probably shouldn't touch them. */

.CodeMirror {
  position: relative;
  overflow: hidden;
  background: white;
}

.CodeMirror-scroll {
  overflow: scroll !important; /* Things will break if this is overridden */
  /* 50px is the magic margin used to hide the element's real scrollbars */
  /* See overflow: hidden in .CodeMirror */
  margin-bottom: -50px; margin-right: -50px;
  padding-bottom: 50px;
  height: 100%;
  outline: none; /* Prevent dragging from highlighting the element */
  position: relative;
}
.CodeMirror-sizer {
  position: relative;
  border-right: 50px solid transparent;
}

/* The fake, visible scrollbars. Used to force redraw during scrolling
   before actual scrolling happens, thus preventing shaking and
   flickering artifacts. */
.CodeMirror-vscrollbar, .CodeMirror-hscrollbar, .CodeMirror-scrollbar-filler, .CodeMirror-gutter-filler {
  position: absolute;
  z-index: 6;
  display: none;
}
.CodeMirror-vscrollbar {
  right: 0; top: 0;
  overflow-x: hidden;
  overflow-y: scroll;
}
.CodeMirror-hscrollbar {
  bottom: 0; left: 0;
  overflow-y: hidden;
  overflow-x: scroll;
}
.CodeMirror-scrollbar-filler {
  right: 0; bottom: 0;
}
.CodeMirror-gutter-filler {
  left: 0; bottom: 0;
}

.CodeMirror-gutters {
  position: absolute; left: 0; top: 0;
  min-height: 100%;
  z-index: 3;
}
.CodeMirror-gutter {
  white-space: normal;
  height: 100%;
  display: inline-block;
  vertical-align: top;
  margin-bottom: -50px;
}
.CodeMirror-gutter-wrapper {
  position: absolute;
  z-index: 4;
  background: none !important;
  border: none !important;
}
.CodeMirror-gutter-background {
  position: absolute;
  top: 0; bottom: 0;
  z-index: 4;
}
.CodeMirror-gutter-elt {
  position: absolute;
  cursor: default;
  z-index: 4;
}
.CodeMirror-gutter-wrapper ::selection { background-color: transparent }
.CodeMirror-gutter-wrapper ::-moz-selection { background-color: transparent }

.CodeMirror-lines {
  cursor: text;
  min-height: 1px; /* prevents collapsing before first draw */
}
.CodeMirror pre.CodeMirror-line,
.CodeMirror pre.CodeMirror-line-like {
  /* Reset some styles that the rest of the page might have set */
  -moz-border-radius: 0; -webkit-border-radius: 0; border-radius: 0;
  border-width: 0;
  background: transparent;
  font-family: inherit;
  font-size: inherit;
  margin: 0;
  white-space: pre;
  word-wrap: normal;
  line-height: inherit;
  color: inherit;
  z-index: 2;
  position: relative;
  overflow: visible;
  -webkit-tap-highlight-color: transparent;
  -webkit-font-variant-ligatures: contextual;
  font-variant-ligatures: contextual;
}
.CodeMirror-wrap pre.CodeMirror-line,
.CodeMirror-wrap pre.CodeMirror-line-like {
  word-wrap: break-word;
  white-space: pre-wrap;
  word-break: normal;
}

.CodeMirror-linebackground {
  position: absolute;
  left: 0; right: 0; top: 0; bottom: 0;
  z-index: 0;
}

.CodeMirror-linewidget {
  position: relative;
  z-index: 2;
  padding: 0.1px; /* Force widget margins to stay inside of the container */
}

.CodeMirror-widget {}

.CodeMirror-rtl pre { direction: rtl; }

.CodeMirror-code {
  outline: none;
}

/* Force content-box sizing for the elements where we expect it */
.CodeMirror-scroll,
.CodeMirror-sizer,
.CodeMirror-gutter,
.CodeMirror-gutters,
.CodeMirror-linenumber {
  -moz-box-sizing: content-box;
  box-sizing: content-box;
}

.CodeMirror-measure {
  position: absolute;
  width: 100%;
  height: 0;
  overflow: hidden;
  visibility: hidden;
}

.CodeMirror-cursor {
  position: absolute;
  pointer-events: none;
}
.CodeMirror-measure pre { position: static; }

div.CodeMirror-cursors {
  visibility: hidden;
  position: relative;
  z-index: 3;
}
div.CodeMirror-dragcursors {
  visibility: visible;
}

.CodeMirror-focused div.CodeMirror-cursors {
  visibility: visible;
}

.CodeMirror-selected { background: #d9d9d9; }
.CodeMirror-focused .CodeMirror-selected { background: #d7d4f0; }
.CodeMirror-crosshair { cursor: crosshair; }
.CodeMirror-line::selection, .CodeMirror-line > span::selection, .CodeMirror-line > span > span::selection { background: #d7d4f0; }
.CodeMirror-line::-moz-selection, .CodeMirror-line > span::-moz-selection, .CodeMirror-line > span > span::-moz-selection { background: #d7d4f0; }

.cm-searching {
  background-color: #ffa;
  background-color: rgba(255, 255, 0, .4);
}

/* Used to force a border model for a node */
.cm-force-border { padding-right: .1px; }

@media print {
  /* Hide the cursor when printing */
  .CodeMirror div.CodeMirror-cursors {
    visibility: hidden;
  }
}

/* See issue #2901 */
.cm-tab-wrap-hack:after { content: ''; }

/* Help users use markselection to safely style text background */
span.CodeMirror-selectedtext { background: none; }

.CodeMirror-dialog {
  position: absolute;
  left: 0; right: 0;
  background: inherit;
  z-index: 15;
  padding: .1em .8em;
  overflow: hidden;
  color: inherit;
}

.CodeMirror-dialog-top {
  border-bottom: 1px solid #eee;
  top: 0;
}

.CodeMirror-dialog-bottom {
  border-top: 1px solid #eee;
  bottom: 0;
}

.CodeMirror-dialog input {
  border: none;
  outline: none;
  background: transparent;
  width: 20em;
  color: inherit;
  font-family: monospace;
}

.CodeMirror-dialog button {
  font-size: 70%;
}

.CodeMirror-foldmarker {
  color: blue;
  text-shadow: #b9f 1px 1px 2px, #b9f -1px -1px 2px, #b9f 1px -1px 2px, #b9f -1px 1px 2px;
  font-family: arial;
  line-height: .3;
  cursor: pointer;
}
.CodeMirror-foldgutter {
  width: .7em;
}
.CodeMirror-foldgutter-open,
.CodeMirror-foldgutter-folded {
  cursor: pointer;
}
.CodeMirror-foldgutter-open:after {
  content: "\25BE";
}
.CodeMirror-foldgutter-folded:after {
  content: "\25B8";
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.CodeMirror {
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  border: 0;
  border-radius: 0;
  height: auto;
  /* Changed to auto to autogrow */
}

.CodeMirror pre {
  padding: 0 var(--jp-code-padding);
}

.jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-dialog {
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

/* This causes https://github.com/jupyter/jupyterlab/issues/522 */
/* May not cause it not because we changed it! */
.CodeMirror-lines {
  padding: var(--jp-code-padding) 0;
}

.CodeMirror-linenumber {
  padding: 0 8px;
}

.jp-CodeMirrorEditor {
  cursor: text;
}

.jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
  border-left: var(--jp-code-cursor-width0) solid var(--jp-editor-cursor-color);
}

/* When zoomed out 67% and 33% on a screen of 1440 width x 900 height */
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
    border-left: var(--jp-code-cursor-width1) solid
      var(--jp-editor-cursor-color);
  }
}

/* When zoomed out less than 33% */
@media screen and (min-width: 4320px) {
  .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
    border-left: var(--jp-code-cursor-width2) solid
      var(--jp-editor-cursor-color);
  }
}

.CodeMirror.jp-mod-readOnly .CodeMirror-cursor {
  display: none;
}

.CodeMirror-gutters {
  border-right: 1px solid var(--jp-border-color2);
  background-color: var(--jp-layout-color0);
}

.jp-CollaboratorCursor {
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: none;
  border-bottom: 3px solid;
  background-clip: content-box;
  margin-left: -5px;
  margin-right: -5px;
}

.CodeMirror-selectedtext.cm-searching {
  background-color: var(--jp-search-selected-match-background-color) !important;
  color: var(--jp-search-selected-match-color) !important;
}

.cm-searching {
  background-color: var(
    --jp-search-unselected-match-background-color
  ) !important;
  color: var(--jp-search-unselected-match-color) !important;
}

.CodeMirror-focused .CodeMirror-selected {
  background-color: var(--jp-editor-selected-focused-background);
}

.CodeMirror-selected {
  background-color: var(--jp-editor-selected-background);
}

.jp-CollaboratorCursor-hover {
  position: absolute;
  z-index: 1;
  transform: translateX(-50%);
  color: white;
  border-radius: 3px;
  padding-left: 4px;
  padding-right: 4px;
  padding-top: 1px;
  padding-bottom: 1px;
  text-align: center;
  font-size: var(--jp-ui-font-size1);
  white-space: nowrap;
}

.jp-CodeMirror-ruler {
  border-left: 1px dashed var(--jp-border-color2);
}

/**
 * Here is our jupyter theme for CodeMirror syntax highlighting
 * This is used in our marked.js syntax highlighting and CodeMirror itself
 * The string "jupyter" is set in ../codemirror/widget.DEFAULT_CODEMIRROR_THEME
 * This came from the classic notebook, which came form highlight.js/GitHub
 */

/**
 * CodeMirror themes are handling the background/color in this way. This works
 * fine for CodeMirror editors outside the notebook, but the notebook styles
 * these things differently.
 */
.CodeMirror.cm-s-jupyter {
  background: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

/* In the notebook, we want this styling to be handled by its container */
.jp-CodeConsole .CodeMirror.cm-s-jupyter,
.jp-Notebook .CodeMirror.cm-s-jupyter {
  background: transparent;
}

.cm-s-jupyter .CodeMirror-cursor {
  border-left: var(--jp-code-cursor-width0) solid var(--jp-editor-cursor-color);
}
.cm-s-jupyter span.cm-keyword {
  color: var(--jp-mirror-editor-keyword-color);
  font-weight: bold;
}
.cm-s-jupyter span.cm-atom {
  color: var(--jp-mirror-editor-atom-color);
}
.cm-s-jupyter span.cm-number {
  color: var(--jp-mirror-editor-number-color);
}
.cm-s-jupyter span.cm-def {
  color: var(--jp-mirror-editor-def-color);
}
.cm-s-jupyter span.cm-variable {
  color: var(--jp-mirror-editor-variable-color);
}
.cm-s-jupyter span.cm-variable-2 {
  color: var(--jp-mirror-editor-variable-2-color);
}
.cm-s-jupyter span.cm-variable-3 {
  color: var(--jp-mirror-editor-variable-3-color);
}
.cm-s-jupyter span.cm-punctuation {
  color: var(--jp-mirror-editor-punctuation-color);
}
.cm-s-jupyter span.cm-property {
  color: var(--jp-mirror-editor-property-color);
}
.cm-s-jupyter span.cm-operator {
  color: var(--jp-mirror-editor-operator-color);
  font-weight: bold;
}
.cm-s-jupyter span.cm-comment {
  color: var(--jp-mirror-editor-comment-color);
  font-style: italic;
}
.cm-s-jupyter span.cm-string {
  color: var(--jp-mirror-editor-string-color);
}
.cm-s-jupyter span.cm-string-2 {
  color: var(--jp-mirror-editor-string-2-color);
}
.cm-s-jupyter span.cm-meta {
  color: var(--jp-mirror-editor-meta-color);
}
.cm-s-jupyter span.cm-qualifier {
  color: var(--jp-mirror-editor-qualifier-color);
}
.cm-s-jupyter span.cm-builtin {
  color: var(--jp-mirror-editor-builtin-color);
}
.cm-s-jupyter span.cm-bracket {
  color: var(--jp-mirror-editor-bracket-color);
}
.cm-s-jupyter span.cm-tag {
  color: var(--jp-mirror-editor-tag-color);
}
.cm-s-jupyter span.cm-attribute {
  color: var(--jp-mirror-editor-attribute-color);
}
.cm-s-jupyter span.cm-header {
  color: var(--jp-mirror-editor-header-color);
}
.cm-s-jupyter span.cm-quote {
  color: var(--jp-mirror-editor-quote-color);
}
.cm-s-jupyter span.cm-link {
  color: var(--jp-mirror-editor-link-color);
}
.cm-s-jupyter span.cm-error {
  color: var(--jp-mirror-editor-error-color);
}
.cm-s-jupyter span.cm-hr {
  color: #999;
}

.cm-s-jupyter span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}

.cm-s-jupyter .CodeMirror-activeline-background,
.cm-s-jupyter .CodeMirror-gutter {
  background-color: var(--jp-layout-color2);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| RenderedText
|----------------------------------------------------------------------------*/

:root {
  /* This is the padding value to fill the gaps between lines containing spans with background color. */
  --jp-private-code-span-padding: calc(
    (var(--jp-code-line-height) - 1) * var(--jp-code-font-size) / 2
  );
}

.jp-RenderedText {
  text-align: left;
  padding-left: var(--jp-code-padding);
  line-height: var(--jp-code-line-height);
  font-family: var(--jp-code-font-family);
}

.jp-RenderedText pre,
.jp-RenderedJavaScript pre,
.jp-RenderedHTMLCommon pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
  border: none;
  margin: 0px;
  padding: 0px;
}

.jp-RenderedText pre a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}
.jp-RenderedText pre a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}
.jp-RenderedText pre a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* console foregrounds and backgrounds */
.jp-RenderedText pre .ansi-black-fg {
  color: #3e424d;
}
.jp-RenderedText pre .ansi-red-fg {
  color: #e75c58;
}
.jp-RenderedText pre .ansi-green-fg {
  color: #00a250;
}
.jp-RenderedText pre .ansi-yellow-fg {
  color: #ddb62b;
}
.jp-RenderedText pre .ansi-blue-fg {
  color: #208ffb;
}
.jp-RenderedText pre .ansi-magenta-fg {
  color: #d160c4;
}
.jp-RenderedText pre .ansi-cyan-fg {
  color: #60c6c8;
}
.jp-RenderedText pre .ansi-white-fg {
  color: #c5c1b4;
}

.jp-RenderedText pre .ansi-black-bg {
  background-color: #3e424d;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-red-bg {
  background-color: #e75c58;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-green-bg {
  background-color: #00a250;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-yellow-bg {
  background-color: #ddb62b;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-blue-bg {
  background-color: #208ffb;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-magenta-bg {
  background-color: #d160c4;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-cyan-bg {
  background-color: #60c6c8;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-white-bg {
  background-color: #c5c1b4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-black-intense-fg {
  color: #282c36;
}
.jp-RenderedText pre .ansi-red-intense-fg {
  color: #b22b31;
}
.jp-RenderedText pre .ansi-green-intense-fg {
  color: #007427;
}
.jp-RenderedText pre .ansi-yellow-intense-fg {
  color: #b27d12;
}
.jp-RenderedText pre .ansi-blue-intense-fg {
  color: #0065ca;
}
.jp-RenderedText pre .ansi-magenta-intense-fg {
  color: #a03196;
}
.jp-RenderedText pre .ansi-cyan-intense-fg {
  color: #258f8f;
}
.jp-RenderedText pre .ansi-white-intense-fg {
  color: #a1a6b2;
}

.jp-RenderedText pre .ansi-black-intense-bg {
  background-color: #282c36;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-red-intense-bg {
  background-color: #b22b31;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-green-intense-bg {
  background-color: #007427;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-yellow-intense-bg {
  background-color: #b27d12;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-blue-intense-bg {
  background-color: #0065ca;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-magenta-intense-bg {
  background-color: #a03196;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-cyan-intense-bg {
  background-color: #258f8f;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-white-intense-bg {
  background-color: #a1a6b2;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-default-inverse-fg {
  color: var(--jp-ui-inverse-font-color0);
}
.jp-RenderedText pre .ansi-default-inverse-bg {
  background-color: var(--jp-inverse-layout-color0);
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-bold {
  font-weight: bold;
}
.jp-RenderedText pre .ansi-underline {
  text-decoration: underline;
}

.jp-RenderedText[data-mime-type='application/vnd.jupyter.stderr'] {
  background: var(--jp-rendermime-error-background);
  padding-top: var(--jp-code-padding);
}

/*-----------------------------------------------------------------------------
| RenderedLatex
|----------------------------------------------------------------------------*/

.jp-RenderedLatex {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
}

/* Left-justify outputs.*/
.jp-OutputArea-output.jp-RenderedLatex {
  padding: var(--jp-code-padding);
  text-align: left;
}

/*-----------------------------------------------------------------------------
| RenderedHTML
|----------------------------------------------------------------------------*/

.jp-RenderedHTMLCommon {
  color: var(--jp-content-font-color1);
  font-family: var(--jp-content-font-family);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
  /* Give a bit more R padding on Markdown text to keep line lengths reasonable */
  padding-right: 20px;
}

.jp-RenderedHTMLCommon em {
  font-style: italic;
}

.jp-RenderedHTMLCommon strong {
  font-weight: bold;
}

.jp-RenderedHTMLCommon u {
  text-decoration: underline;
}

.jp-RenderedHTMLCommon a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* Headings */

.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4,
.jp-RenderedHTMLCommon h5,
.jp-RenderedHTMLCommon h6 {
  line-height: var(--jp-content-heading-line-height);
  font-weight: var(--jp-content-heading-font-weight);
  font-style: normal;
  margin: var(--jp-content-heading-margin-top) 0
    var(--jp-content-heading-margin-bottom) 0;
}

.jp-RenderedHTMLCommon h1:first-child,
.jp-RenderedHTMLCommon h2:first-child,
.jp-RenderedHTMLCommon h3:first-child,
.jp-RenderedHTMLCommon h4:first-child,
.jp-RenderedHTMLCommon h5:first-child,
.jp-RenderedHTMLCommon h6:first-child {
  margin-top: calc(0.5 * var(--jp-content-heading-margin-top));
}

.jp-RenderedHTMLCommon h1:last-child,
.jp-RenderedHTMLCommon h2:last-child,
.jp-RenderedHTMLCommon h3:last-child,
.jp-RenderedHTMLCommon h4:last-child,
.jp-RenderedHTMLCommon h5:last-child,
.jp-RenderedHTMLCommon h6:last-child {
  margin-bottom: calc(0.5 * var(--jp-content-heading-margin-bottom));
}

.jp-RenderedHTMLCommon h1 {
  font-size: var(--jp-content-font-size5);
}

.jp-RenderedHTMLCommon h2 {
  font-size: var(--jp-content-font-size4);
}

.jp-RenderedHTMLCommon h3 {
  font-size: var(--jp-content-font-size3);
}

.jp-RenderedHTMLCommon h4 {
  font-size: var(--jp-content-font-size2);
}

.jp-RenderedHTMLCommon h5 {
  font-size: var(--jp-content-font-size1);
}

.jp-RenderedHTMLCommon h6 {
  font-size: var(--jp-content-font-size0);
}

/* Lists */

.jp-RenderedHTMLCommon ul:not(.list-inline),
.jp-RenderedHTMLCommon ol:not(.list-inline) {
  padding-left: 2em;
}

.jp-RenderedHTMLCommon ul {
  list-style: disc;
}

.jp-RenderedHTMLCommon ul ul {
  list-style: square;
}

.jp-RenderedHTMLCommon ul ul ul {
  list-style: circle;
}

.jp-RenderedHTMLCommon ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol ol {
  list-style: upper-alpha;
}

.jp-RenderedHTMLCommon ol ol ol {
  list-style: lower-alpha;
}

.jp-RenderedHTMLCommon ol ol ol ol {
  list-style: lower-roman;
}

.jp-RenderedHTMLCommon ol ol ol ol ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol,
.jp-RenderedHTMLCommon ul {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon ul ul,
.jp-RenderedHTMLCommon ul ol,
.jp-RenderedHTMLCommon ol ul,
.jp-RenderedHTMLCommon ol ol {
  margin-bottom: 0em;
}

.jp-RenderedHTMLCommon hr {
  color: var(--jp-border-color2);
  background-color: var(--jp-border-color1);
  margin-top: 1em;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon > pre {
  margin: 1.5em 2em;
}

.jp-RenderedHTMLCommon pre,
.jp-RenderedHTMLCommon code {
  border: 0;
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  line-height: var(--jp-code-line-height);
  padding: 0;
  white-space: pre-wrap;
}

.jp-RenderedHTMLCommon :not(pre) > code {
  background-color: var(--jp-layout-color2);
  padding: 1px 5px;
}

/* Tables */

.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
  border-spacing: 0;
  border: none;
  color: var(--jp-ui-font-color1);
  font-size: 12px;
  table-layout: fixed;
  margin-left: auto;
  margin-right: auto;
}

.jp-RenderedHTMLCommon thead {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  vertical-align: bottom;
}

.jp-RenderedHTMLCommon td,
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon tr {
  vertical-align: middle;
  padding: 0.5em 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}

.jp-RenderedMarkdown.jp-RenderedHTMLCommon td,
.jp-RenderedMarkdown.jp-RenderedHTMLCommon th {
  max-width: none;
}

:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon td,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon th,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon tr {
  text-align: right;
}

.jp-RenderedHTMLCommon th {
  font-weight: bold;
}

.jp-RenderedHTMLCommon tbody tr:nth-child(odd) {
  background: var(--jp-layout-color0);
}

.jp-RenderedHTMLCommon tbody tr:nth-child(even) {
  background: var(--jp-rendermime-table-row-background);
}

.jp-RenderedHTMLCommon tbody tr:hover {
  background: var(--jp-rendermime-table-row-hover-background);
}

.jp-RenderedHTMLCommon table {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon p {
  text-align: left;
  margin: 0px;
}

.jp-RenderedHTMLCommon p {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon img {
  -moz-force-broken-image-icon: 1;
}

/* Restrict to direct children as other images could be nested in other content. */
.jp-RenderedHTMLCommon > img {
  display: block;
  margin-left: 0;
  margin-right: 0;
  margin-bottom: 1em;
}

/* Change color behind transparent images if they need it... */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-light-background {
  background-color: var(--jp-inverse-layout-color1);
}
[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-dark-background {
  background-color: var(--jp-inverse-layout-color1);
}
/* ...or leave it untouched if they don't */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-dark-background {
}
[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-light-background {
}

.jp-RenderedHTMLCommon img,
.jp-RenderedImage img,
.jp-RenderedHTMLCommon svg,
.jp-RenderedSVG svg {
  max-width: 100%;
  height: auto;
}

.jp-RenderedHTMLCommon img.jp-mod-unconfined,
.jp-RenderedImage img.jp-mod-unconfined,
.jp-RenderedHTMLCommon svg.jp-mod-unconfined,
.jp-RenderedSVG svg.jp-mod-unconfined {
  max-width: none;
}

.jp-RenderedHTMLCommon .alert {
  padding: var(--jp-notebook-padding);
  border: var(--jp-border-width) solid transparent;
  border-radius: var(--jp-border-radius);
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon .alert-info {
  color: var(--jp-info-color0);
  background-color: var(--jp-info-color3);
  border-color: var(--jp-info-color2);
}
.jp-RenderedHTMLCommon .alert-info hr {
  border-color: var(--jp-info-color3);
}
.jp-RenderedHTMLCommon .alert-info > p:last-child,
.jp-RenderedHTMLCommon .alert-info > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-warning {
  color: var(--jp-warn-color0);
  background-color: var(--jp-warn-color3);
  border-color: var(--jp-warn-color2);
}
.jp-RenderedHTMLCommon .alert-warning hr {
  border-color: var(--jp-warn-color3);
}
.jp-RenderedHTMLCommon .alert-warning > p:last-child,
.jp-RenderedHTMLCommon .alert-warning > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-success {
  color: var(--jp-success-color0);
  background-color: var(--jp-success-color3);
  border-color: var(--jp-success-color2);
}
.jp-RenderedHTMLCommon .alert-success hr {
  border-color: var(--jp-success-color3);
}
.jp-RenderedHTMLCommon .alert-success > p:last-child,
.jp-RenderedHTMLCommon .alert-success > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-danger {
  color: var(--jp-error-color0);
  background-color: var(--jp-error-color3);
  border-color: var(--jp-error-color2);
}
.jp-RenderedHTMLCommon .alert-danger hr {
  border-color: var(--jp-error-color3);
}
.jp-RenderedHTMLCommon .alert-danger > p:last-child,
.jp-RenderedHTMLCommon .alert-danger > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon blockquote {
  margin: 1em 2em;
  padding: 0 1em;
  border-left: 5px solid var(--jp-border-color2);
}

a.jp-InternalAnchorLink {
  visibility: hidden;
  margin-left: 8px;
  color: var(--md-blue-800);
}

h1:hover .jp-InternalAnchorLink,
h2:hover .jp-InternalAnchorLink,
h3:hover .jp-InternalAnchorLink,
h4:hover .jp-InternalAnchorLink,
h5:hover .jp-InternalAnchorLink,
h6:hover .jp-InternalAnchorLink {
  visibility: visible;
}

.jp-RenderedHTMLCommon kbd {
  background-color: var(--jp-rendermime-table-row-background);
  border: 1px solid var(--jp-border-color0);
  border-bottom-color: var(--jp-border-color2);
  border-radius: 3px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
  display: inline-block;
  font-size: 0.8em;
  line-height: 1em;
  padding: 0.2em 0.5em;
}

/* Most direct children of .jp-RenderedHTMLCommon have a margin-bottom of 1.0.
 * At the bottom of cells this is a bit too much as there is also spacing
 * between cells. Going all the way to 0 gets too tight between markdown and
 * code cells.
 */
.jp-RenderedHTMLCommon > *:last-child {
  margin-bottom: 0.5em;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MimeDocument {
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-filebrowser-button-height: 28px;
  --jp-private-filebrowser-button-width: 48px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FileBrowser {
  display: flex;
  flex-direction: column;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  border-bottom: none;
  height: auto;
  margin: var(--jp-toolbar-header-margin);
  box-shadow: none;
}

.jp-BreadCrumbs {
  flex: 0 0 auto;
  margin: 8px 12px 8px 12px;
}

.jp-BreadCrumbs-item {
  margin: 0px 2px;
  padding: 0px 2px;
  border-radius: var(--jp-border-radius);
  cursor: pointer;
}

.jp-BreadCrumbs-item:hover {
  background-color: var(--jp-layout-color2);
}

.jp-BreadCrumbs-item:first-child {
  margin-left: 0px;
}

.jp-BreadCrumbs-item.jp-mod-dropTarget {
  background-color: var(--jp-brand-color2);
  opacity: 0.7;
}

/*-----------------------------------------------------------------------------
| Buttons
|----------------------------------------------------------------------------*/

.jp-FileBrowser-toolbar.jp-Toolbar {
  padding: 0px;
  margin: 8px 12px 0px 12px;
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  justify-content: flex-start;
}

.jp-FileBrowser-toolbar.jp-Toolbar .jp-Toolbar-item {
  flex: 0 0 auto;
  padding-left: 0px;
  padding-right: 2px;
}

.jp-FileBrowser-toolbar.jp-Toolbar .jp-ToolbarButtonComponent {
  width: 40px;
}

.jp-FileBrowser-toolbar.jp-Toolbar
  .jp-Toolbar-item:first-child
  .jp-ToolbarButtonComponent {
  width: 72px;
  background: var(--jp-brand-color1);
}

.jp-FileBrowser-toolbar.jp-Toolbar
  .jp-Toolbar-item:first-child
  .jp-ToolbarButtonComponent
  .jp-icon3 {
  fill: white;
}

/*-----------------------------------------------------------------------------
| Other styles
|----------------------------------------------------------------------------*/

.jp-FileDialog.jp-mod-conflict input {
  color: red;
}

.jp-FileDialog .jp-new-name-title {
  margin-top: 12px;
}

.jp-LastModified-hidden {
  display: none;
}

.jp-FileBrowser-filterBox {
  padding: 0px;
  flex: 0 0 auto;
  margin: 8px 12px 0px 12px;
}

/*-----------------------------------------------------------------------------
| DirListing
|----------------------------------------------------------------------------*/

.jp-DirListing {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  outline: 0;
}

.jp-DirListing-header {
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  overflow: hidden;
  border-top: var(--jp-border-width) solid var(--jp-border-color2);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
}

.jp-DirListing-headerItem {
  padding: 4px 12px 2px 12px;
  font-weight: 500;
}

.jp-DirListing-headerItem:hover {
  background: var(--jp-layout-color2);
}

.jp-DirListing-headerItem.jp-id-name {
  flex: 1 0 84px;
}

.jp-DirListing-headerItem.jp-id-modified {
  flex: 0 0 112px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-id-narrow {
  display: none;
  flex: 0 0 5px;
  padding: 4px 4px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
  color: var(--jp-border-color2);
}

.jp-DirListing-narrow .jp-id-narrow {
  display: block;
}

.jp-DirListing-narrow .jp-id-modified,
.jp-DirListing-narrow .jp-DirListing-itemModified {
  display: none;
}

.jp-DirListing-headerItem.jp-mod-selected {
  font-weight: 600;
}

/* increase specificity to override bundled default */
.jp-DirListing-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-content mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

/* Style the directory listing content when a user drops a file to upload */
.jp-DirListing.jp-mod-native-drop .jp-DirListing-content {
  outline: 5px dashed rgba(128, 128, 128, 0.5);
  outline-offset: -10px;
  cursor: copy;
}

.jp-DirListing-item {
  display: flex;
  flex-direction: row;
  padding: 4px 12px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-DirListing-item[data-is-dot] {
  opacity: 75%;
}

.jp-DirListing-item.jp-mod-selected {
  color: white;
  background: var(--jp-brand-color1);
}

.jp-DirListing-item.jp-mod-dropTarget {
  background: var(--jp-brand-color3);
}

.jp-DirListing-item:hover:not(.jp-mod-selected) {
  background: var(--jp-layout-color2);
}

.jp-DirListing-itemIcon {
  flex: 0 0 20px;
  margin-right: 4px;
}

.jp-DirListing-itemText {
  flex: 1 0 64px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  user-select: none;
}

.jp-DirListing-itemModified {
  flex: 0 0 125px;
  text-align: right;
}

.jp-DirListing-editor {
  flex: 1 0 64px;
  outline: none;
  border: none;
}

.jp-DirListing-item.jp-mod-running .jp-DirListing-itemIcon:before {
  color: limegreen;
  content: '\25CF';
  font-size: 8px;
  position: absolute;
  left: -8px;
}

.jp-DirListing-item.lm-mod-drag-image,
.jp-DirListing-item.jp-mod-selected.lm-mod-drag-image {
  font-size: var(--jp-ui-font-size1);
  padding-left: 4px;
  margin-left: 4px;
  width: 160px;
  background-color: var(--jp-ui-inverse-font-color2);
  box-shadow: var(--jp-elevation-z2);
  border-radius: 0px;
  color: var(--jp-ui-font-color1);
  transform: translateX(-40%) translateY(-58%);
}

.jp-DirListing-deadSpace {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-Document {
  min-width: 120px;
  min-height: 120px;
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
}

/*-----------------------------------------------------------------------------
| Main OutputArea
| OutputArea has a list of Outputs
|----------------------------------------------------------------------------*/

.jp-OutputArea {
  overflow-y: auto;
}

.jp-OutputArea-child {
  display: flex;
  flex-direction: row;
}

.jp-OutputPrompt {
  flex: 0 0 var(--jp-cell-prompt-width);
  color: var(--jp-cell-outprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);
  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-OutputArea-output {
  height: auto;
  overflow: auto;
  user-select: text;
  -moz-user-select: text;
  -webkit-user-select: text;
  -ms-user-select: text;
}

.jp-OutputArea-child .jp-OutputArea-output {
  flex-grow: 1;
  flex-shrink: 1;
}

/**
 * Isolated output.
 */
.jp-OutputArea-output.jp-mod-isolated {
  width: 100%;
  display: block;
}

/*
When drag events occur, `p-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated {
  position: relative;
}

body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated:before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/* pre */

.jp-OutputArea-output pre {
  border: none;
  margin: 0px;
  padding: 0px;
  overflow-x: auto;
  overflow-y: auto;
  word-break: break-all;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* tables */

.jp-OutputArea-output.jp-RenderedHTMLCommon table {
  margin-left: 0;
  margin-right: 0;
}

/* description lists */

.jp-OutputArea-output dl,
.jp-OutputArea-output dt,
.jp-OutputArea-output dd {
  display: block;
}

.jp-OutputArea-output dl {
  width: 100%;
  overflow: hidden;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dt {
  font-weight: bold;
  float: left;
  width: 20%;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dd {
  float: left;
  width: 80%;
  padding: 0;
  margin: 0;
}

/* Hide the gutter in case of
 *  - nested output areas (e.g. in the case of output widgets)
 *  - mirrored output areas
 */
.jp-OutputArea .jp-OutputArea .jp-OutputArea-prompt {
  display: none;
}

/*-----------------------------------------------------------------------------
| executeResult is added to any Output-result for the display of the object
| returned by a cell
|----------------------------------------------------------------------------*/

.jp-OutputArea-output.jp-OutputArea-executeResult {
  margin-left: 0px;
  flex: 1 1 auto;
}

/* Text output with the Out[] prompt needs a top padding to match the
 * alignment of the Out[] prompt itself.
 */
.jp-OutputArea-executeResult .jp-RenderedText.jp-OutputArea-output {
  padding-top: var(--jp-code-padding);
  border-top: var(--jp-border-width) solid transparent;
}

/*-----------------------------------------------------------------------------
| The Stdin output
|----------------------------------------------------------------------------*/

.jp-OutputArea-stdin {
  line-height: var(--jp-code-line-height);
  padding-top: var(--jp-code-padding);
  display: flex;
}

.jp-Stdin-prompt {
  color: var(--jp-content-font-color0);
  padding-right: var(--jp-code-padding);
  vertical-align: baseline;
  flex: 0 0 auto;
}

.jp-Stdin-input {
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  color: inherit;
  background-color: inherit;
  width: 42%;
  min-width: 200px;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
  flex: 0 0 70%;
}

.jp-Stdin-input:focus {
  box-shadow: none;
}

/*-----------------------------------------------------------------------------
| Output Area View
|----------------------------------------------------------------------------*/

.jp-LinkedOutputView .jp-OutputArea {
  height: 100%;
  display: block;
}

.jp-LinkedOutputView .jp-OutputArea-output:only-child {
  height: 100%;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapser {
  flex: 0 0 var(--jp-cell-collapser-width);
  padding: 0px;
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
  border-radius: var(--jp-border-radius);
  opacity: 1;
}

.jp-Collapser-child {
  display: block;
  width: 100%;
  box-sizing: border-box;
  /* height: 100% doesn't work because the height of its parent is computed from content */
  position: absolute;
  top: 0px;
  bottom: 0px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Header/Footer
|----------------------------------------------------------------------------*/

/* Hidden by zero height by default */
.jp-CellHeader,
.jp-CellFooter {
  height: 0px;
  width: 100%;
  padding: 0px;
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Input
|----------------------------------------------------------------------------*/

/* All input areas */
.jp-InputArea {
  display: flex;
  flex-direction: row;
  overflow: hidden;
}

.jp-InputArea-editor {
  flex: 1 1 auto;
  overflow: hidden;
}

.jp-InputArea-editor {
  /* This is the non-active, default styling */
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0px;
  background: var(--jp-cell-editor-background);
}

.jp-InputPrompt {
  flex: 0 0 var(--jp-cell-prompt-width);
  color: var(--jp-cell-inprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  opacity: var(--jp-cell-prompt-opacity);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);
  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Placeholder {
  display: flex;
  flex-direction: row;
  flex: 1 1 auto;
}

.jp-Placeholder-prompt {
  box-sizing: border-box;
}

.jp-Placeholder-content {
  flex: 1 1 auto;
  border: none;
  background: transparent;
  height: 20px;
  box-sizing: border-box;
}

.jp-Placeholder-content .jp-MoreHorizIcon {
  width: 32px;
  height: 16px;
  border: 1px solid transparent;
  border-radius: var(--jp-border-radius);
}

.jp-Placeholder-content .jp-MoreHorizIcon:hover {
  border: 1px solid var(--jp-border-color1);
  box-shadow: 0px 0px 2px 0px rgba(0, 0, 0, 0.25);
  background-color: var(--jp-layout-color0);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-cell-scrolling-output-offset: 5px;
}

/*-----------------------------------------------------------------------------
| Cell
|----------------------------------------------------------------------------*/

.jp-Cell {
  padding: var(--jp-cell-padding);
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Common input/output
|----------------------------------------------------------------------------*/

.jp-Cell-inputWrapper,
.jp-Cell-outputWrapper {
  display: flex;
  flex-direction: row;
  padding: 0px;
  margin: 0px;
  /* Added to reveal the box-shadow on the input and output collapsers. */
  overflow: visible;
}

/* Only input/output areas inside cells */
.jp-Cell-inputArea,
.jp-Cell-outputArea {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Collapser
|----------------------------------------------------------------------------*/

/* Make the output collapser disappear when there is not output, but do so
 * in a manner that leaves it in the layout and preserves its width.
 */
.jp-Cell.jp-mod-noOutputs .jp-Cell-outputCollapser {
  border: none !important;
  background: transparent !important;
}

.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputCollapser {
  min-height: var(--jp-cell-collapser-min-height);
}

/*-----------------------------------------------------------------------------
| Output
|----------------------------------------------------------------------------*/

/* Put a space between input and output when there IS output */
.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputWrapper {
  margin-top: 5px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea {
  overflow-y: auto;
  max-height: 200px;
  box-shadow: inset 0 0 6px 2px rgba(0, 0, 0, 0.3);
  margin-left: var(--jp-private-cell-scrolling-output-offset);
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  flex: 0 0
    calc(
      var(--jp-cell-prompt-width) -
        var(--jp-private-cell-scrolling-output-offset)
    );
}

/*-----------------------------------------------------------------------------
| CodeCell
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| MarkdownCell
|----------------------------------------------------------------------------*/

.jp-MarkdownOutput {
  flex: 1 1 auto;
  margin-top: 0;
  margin-bottom: 0;
  padding-left: var(--jp-code-padding);
}

.jp-MarkdownOutput.jp-RenderedHTMLCommon {
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-NotebookPanel-toolbar {
  padding: 2px;
}

.jp-Toolbar-item.jp-Notebook-toolbarCellType .jp-select-wrapper.jp-mod-focused {
  border: none;
  box-shadow: none;
}

.jp-Notebook-toolbarCellTypeDropdown select {
  height: 24px;
  font-size: var(--jp-ui-font-size1);
  line-height: 14px;
  border-radius: 0;
  display: block;
}

.jp-Notebook-toolbarCellTypeDropdown span {
  top: 5px !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-notebook-dragImage-width: 304px;
  --jp-private-notebook-dragImage-height: 36px;
  --jp-private-notebook-selected-color: var(--md-blue-400);
  --jp-private-notebook-active-color: var(--md-green-400);
}

/*-----------------------------------------------------------------------------
| Imports
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Notebook
|----------------------------------------------------------------------------*/

.jp-NotebookPanel {
  display: block;
  height: 100%;
}

.jp-NotebookPanel.jp-Document {
  min-width: 240px;
  min-height: 120px;
}

.jp-Notebook {
  padding: var(--jp-notebook-padding);
  outline: none;
  overflow: auto;
  background: var(--jp-layout-color0);
}

.jp-Notebook.jp-mod-scrollPastEnd::after {
  display: block;
  content: '';
  min-height: var(--jp-notebook-scroll-padding);
}

.jp-Notebook .jp-Cell {
  overflow: visible;
}

.jp-Notebook .jp-Cell .jp-InputPrompt {
  cursor: move;
}

/*-----------------------------------------------------------------------------
| Notebook state related styling
|
| The notebook and cells each have states, here are the possibilities:
|
| - Notebook
|   - Command
|   - Edit
| - Cell
|   - None
|   - Active (only one can be active)
|   - Selected (the cells actions are applied to)
|   - Multiselected (when multiple selected, the cursor)
|   - No outputs
|----------------------------------------------------------------------------*/

/* Command or edit modes */

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-InputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-OutputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

/* cell is active */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser {
  background: var(--jp-brand-color1);
}

/* collapser is hovered */
.jp-Notebook .jp-Cell .jp-Collapser:hover {
  box-shadow: var(--jp-elevation-z2);
  background: var(--jp-brand-color1);
  opacity: var(--jp-cell-collapser-not-active-hover-opacity);
}

/* cell is active and collapser is hovered */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser:hover {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/* Command mode */

.jp-Notebook.jp-mod-commandMode .jp-Cell.jp-mod-selected {
  background: var(--jp-notebook-multiselected-color);
}

.jp-Notebook.jp-mod-commandMode
  .jp-Cell.jp-mod-active.jp-mod-selected:not(.jp-mod-multiSelected) {
  background: transparent;
}

/* Edit mode */

.jp-Notebook.jp-mod-editMode .jp-Cell.jp-mod-active .jp-InputArea-editor {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-cell-editor-active-background);
}

/*-----------------------------------------------------------------------------
| Notebook drag and drop
|----------------------------------------------------------------------------*/

.jp-Notebook-cell.jp-mod-dropSource {
  opacity: 0.5;
}

.jp-Notebook-cell.jp-mod-dropTarget,
.jp-Notebook.jp-mod-commandMode
  .jp-Notebook-cell.jp-mod-active.jp-mod-selected.jp-mod-dropTarget {
  border-top-color: var(--jp-private-notebook-selected-color);
  border-top-style: solid;
  border-top-width: 2px;
}

.jp-dragImage {
  display: flex;
  flex-direction: row;
  width: var(--jp-private-notebook-dragImage-width);
  height: var(--jp-private-notebook-dragImage-height);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
  overflow: visible;
}

.jp-dragImage-singlePrompt {
  box-shadow: 2px 2px 4px 0px rgba(0, 0, 0, 0.12);
}

.jp-dragImage .jp-dragImage-content {
  flex: 1 1 auto;
  z-index: 2;
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  line-height: var(--jp-code-line-height);
  padding: var(--jp-code-padding);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background-color);
  color: var(--jp-content-font-color3);
  text-align: left;
  margin: 4px 4px 4px 0px;
}

.jp-dragImage .jp-dragImage-prompt {
  flex: 0 0 auto;
  min-width: 36px;
  color: var(--jp-cell-inprompt-font-color);
  padding: var(--jp-code-padding);
  padding-left: 12px;
  font-family: var(--jp-cell-prompt-font-family);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: 1.9;
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
}

.jp-dragImage-multipleBack {
  z-index: -1;
  position: absolute;
  height: 32px;
  width: 300px;
  top: 8px;
  left: 8px;
  background: var(--jp-layout-color2);
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  box-shadow: 2px 2px 4px 0px rgba(0, 0, 0, 0.12);
}

/*-----------------------------------------------------------------------------
| Cell toolbar
|----------------------------------------------------------------------------*/

.jp-NotebookTools {
  display: block;
  min-width: var(--jp-sidebar-min-width);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
    * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  overflow: auto;
}

.jp-NotebookTools-tool {
  padding: 0px 12px 0 12px;
}

.jp-ActiveCellTool {
  padding: 12px;
  background-color: var(--jp-layout-color1);
  border-top: none !important;
}

.jp-ActiveCellTool .jp-InputArea-prompt {
  flex: 0 0 auto;
  padding-left: 0px;
}

.jp-ActiveCellTool .jp-InputArea-editor {
  flex: 1 1 auto;
  background: var(--jp-cell-editor-background);
  border-color: var(--jp-cell-editor-border-color);
}

.jp-ActiveCellTool .jp-InputArea-editor .CodeMirror {
  background: transparent;
}

.jp-MetadataEditorTool {
  flex-direction: column;
  padding: 12px 0px 12px 0px;
}

.jp-RankedPanel > :not(:first-child) {
  margin-top: 12px;
}

.jp-KeySelector select.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: var(--jp-border-width) solid var(--jp-border-color1);
}

.jp-KeySelector label,
.jp-MetadataEditorTool label {
  line-height: 1.4;
}

.jp-NotebookTools .jp-select-wrapper {
  margin-top: 4px;
  margin-bottom: 0px;
}

.jp-NotebookTools .jp-Collapse {
  margin-top: 16px;
}

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-Notebook {
  --jp-content-font-size1: var(--jp-content-presentation-font-size1);
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}

.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-InputPrompt,
.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-OutputPrompt {
  flex: 0 0 110px;
}

</style>

    <style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
The following CSS variables define the main, public API for styling JupyterLab.
These variables should be used by all plugins wherever possible. In other
words, plugins should not define custom colors, sizes, etc unless absolutely
necessary. This enables users to change the visual theme of JupyterLab
by changing these variables.

Many variables appear in an ordered sequence (0,1,2,3). These sequences
are designed to work well together, so for example, `--jp-border-color1` should
be used with `--jp-layout-color1`. The numbers have the following meanings:

* 0: super-primary, reserved for special emphasis
* 1: primary, most important under normal situations
* 2: secondary, next most important under normal situations
* 3: tertiary, next most important under normal situations

Throughout JupyterLab, we are mostly following principles from Google's
Material Design when selecting colors. We are not, however, following
all of MD as it is not optimized for dense, information rich UIs.
*/

:root {
  /* Elevation
   *
   * We style box-shadows using Material Design's idea of elevation. These particular numbers are taken from here:
   *
   * https://github.com/material-components/material-components-web
   * https://material-components-web.appspot.com/elevation.html
   */

  --jp-shadow-base-lightness: 0;
  --jp-shadow-umbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.2
  );
  --jp-shadow-penumbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.14
  );
  --jp-shadow-ambient-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.12
  );
  --jp-elevation-z0: none;
  --jp-elevation-z1: 0px 2px 1px -1px var(--jp-shadow-umbra-color),
    0px 1px 1px 0px var(--jp-shadow-penumbra-color),
    0px 1px 3px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z2: 0px 3px 1px -2px var(--jp-shadow-umbra-color),
    0px 2px 2px 0px var(--jp-shadow-penumbra-color),
    0px 1px 5px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z4: 0px 2px 4px -1px var(--jp-shadow-umbra-color),
    0px 4px 5px 0px var(--jp-shadow-penumbra-color),
    0px 1px 10px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z6: 0px 3px 5px -1px var(--jp-shadow-umbra-color),
    0px 6px 10px 0px var(--jp-shadow-penumbra-color),
    0px 1px 18px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z8: 0px 5px 5px -3px var(--jp-shadow-umbra-color),
    0px 8px 10px 1px var(--jp-shadow-penumbra-color),
    0px 3px 14px 2px var(--jp-shadow-ambient-color);
  --jp-elevation-z12: 0px 7px 8px -4px var(--jp-shadow-umbra-color),
    0px 12px 17px 2px var(--jp-shadow-penumbra-color),
    0px 5px 22px 4px var(--jp-shadow-ambient-color);
  --jp-elevation-z16: 0px 8px 10px -5px var(--jp-shadow-umbra-color),
    0px 16px 24px 2px var(--jp-shadow-penumbra-color),
    0px 6px 30px 5px var(--jp-shadow-ambient-color);
  --jp-elevation-z20: 0px 10px 13px -6px var(--jp-shadow-umbra-color),
    0px 20px 31px 3px var(--jp-shadow-penumbra-color),
    0px 8px 38px 7px var(--jp-shadow-ambient-color);
  --jp-elevation-z24: 0px 11px 15px -7px var(--jp-shadow-umbra-color),
    0px 24px 38px 3px var(--jp-shadow-penumbra-color),
    0px 9px 46px 8px var(--jp-shadow-ambient-color);

  /* Borders
   *
   * The following variables, specify the visual styling of borders in JupyterLab.
   */

  --jp-border-width: 1px;
  --jp-border-color0: var(--md-grey-400);
  --jp-border-color1: var(--md-grey-400);
  --jp-border-color2: var(--md-grey-300);
  --jp-border-color3: var(--md-grey-200);
  --jp-border-radius: 2px;

  /* UI Fonts
   *
   * The UI font CSS variables are used for the typography all of the JupyterLab
   * user interface elements that are not directly user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-ui-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: 0.83333em;
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: 1.2em;
  --jp-ui-font-size3: 1.44em;

  --jp-ui-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica,
    Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';

  /*
   * Use these font colors against the corresponding main layout colors.
   * In a light theme, these go from dark to light.
   */

  /* Defaults use Material Design specification */
  --jp-ui-font-color0: rgba(0, 0, 0, 1);
  --jp-ui-font-color1: rgba(0, 0, 0, 0.87);
  --jp-ui-font-color2: rgba(0, 0, 0, 0.54);
  --jp-ui-font-color3: rgba(0, 0, 0, 0.38);

  /*
   * Use these against the brand/accent/warn/error colors.
   * These will typically go from light to darker, in both a dark and light theme.
   */

  --jp-ui-inverse-font-color0: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color1: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color2: rgba(255, 255, 255, 0.7);
  --jp-ui-inverse-font-color3: rgba(255, 255, 255, 0.5);

  /* Content Fonts
   *
   * Content font variables are used for typography of user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-content-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-content-line-height: 1.6;
  --jp-content-font-scale-factor: 1.2;
  --jp-content-font-size0: 0.83333em;
  --jp-content-font-size1: 14px; /* Base font size */
  --jp-content-font-size2: 1.2em;
  --jp-content-font-size3: 1.44em;
  --jp-content-font-size4: 1.728em;
  --jp-content-font-size5: 2.0736em;

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-content-presentation-font-size1: 17px;

  --jp-content-heading-line-height: 1;
  --jp-content-heading-margin-top: 1.2em;
  --jp-content-heading-margin-bottom: 0.8em;
  --jp-content-heading-font-weight: 500;

  /* Defaults use Material Design specification */
  --jp-content-font-color0: rgba(0, 0, 0, 1);
  --jp-content-font-color1: rgba(0, 0, 0, 0.87);
  --jp-content-font-color2: rgba(0, 0, 0, 0.54);
  --jp-content-font-color3: rgba(0, 0, 0, 0.38);

  --jp-content-link-color: var(--md-blue-700);

  --jp-content-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
    Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji',
    'Segoe UI Symbol';

  /*
   * Code Fonts
   *
   * Code font variables are used for typography of code and other monospaces content.
   */

  --jp-code-font-size: 13px;
  --jp-code-line-height: 1.3077; /* 17px for 13px base */
  --jp-code-padding: 5px; /* 5px for 13px base, codemirror highlighting needs integer px value */
  --jp-code-font-family-default: Menlo, Consolas, 'DejaVu Sans Mono', monospace;
  --jp-code-font-family: var(--jp-code-font-family-default);

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-code-presentation-font-size: 16px;

  /* may need to tweak cursor width if you change font size */
  --jp-code-cursor-width0: 1.4px;
  --jp-code-cursor-width1: 2px;
  --jp-code-cursor-width2: 4px;

  /* Layout
   *
   * The following are the main layout colors use in JupyterLab. In a light
   * theme these would go from light to dark.
   */

  --jp-layout-color0: white;
  --jp-layout-color1: white;
  --jp-layout-color2: var(--md-grey-200);
  --jp-layout-color3: var(--md-grey-400);
  --jp-layout-color4: var(--md-grey-600);

  /* Inverse Layout
   *
   * The following are the inverse layout colors use in JupyterLab. In a light
   * theme these would go from dark to light.
   */

  --jp-inverse-layout-color0: #111111;
  --jp-inverse-layout-color1: var(--md-grey-900);
  --jp-inverse-layout-color2: var(--md-grey-800);
  --jp-inverse-layout-color3: var(--md-grey-700);
  --jp-inverse-layout-color4: var(--md-grey-600);

  /* Brand/accent */

  --jp-brand-color0: var(--md-blue-700);
  --jp-brand-color1: var(--md-blue-500);
  --jp-brand-color2: var(--md-blue-300);
  --jp-brand-color3: var(--md-blue-100);
  --jp-brand-color4: var(--md-blue-50);

  --jp-accent-color0: var(--md-green-700);
  --jp-accent-color1: var(--md-green-500);
  --jp-accent-color2: var(--md-green-300);
  --jp-accent-color3: var(--md-green-100);

  /* State colors (warn, error, success, info) */

  --jp-warn-color0: var(--md-orange-700);
  --jp-warn-color1: var(--md-orange-500);
  --jp-warn-color2: var(--md-orange-300);
  --jp-warn-color3: var(--md-orange-100);

  --jp-error-color0: var(--md-red-700);
  --jp-error-color1: var(--md-red-500);
  --jp-error-color2: var(--md-red-300);
  --jp-error-color3: var(--md-red-100);

  --jp-success-color0: var(--md-green-700);
  --jp-success-color1: var(--md-green-500);
  --jp-success-color2: var(--md-green-300);
  --jp-success-color3: var(--md-green-100);

  --jp-info-color0: var(--md-cyan-700);
  --jp-info-color1: var(--md-cyan-500);
  --jp-info-color2: var(--md-cyan-300);
  --jp-info-color3: var(--md-cyan-100);

  /* Cell specific styles */

  --jp-cell-padding: 5px;

  --jp-cell-collapser-width: 8px;
  --jp-cell-collapser-min-height: 20px;
  --jp-cell-collapser-not-active-hover-opacity: 0.6;

  --jp-cell-editor-background: var(--md-grey-100);
  --jp-cell-editor-border-color: var(--md-grey-300);
  --jp-cell-editor-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-cell-editor-active-background: var(--jp-layout-color0);
  --jp-cell-editor-active-border-color: var(--jp-brand-color1);

  --jp-cell-prompt-width: 64px;
  --jp-cell-prompt-font-family: var(--jp-code-font-family-default);
  --jp-cell-prompt-letter-spacing: 0px;
  --jp-cell-prompt-opacity: 1;
  --jp-cell-prompt-not-active-opacity: 0.5;
  --jp-cell-prompt-not-active-font-color: var(--md-grey-700);
  /* A custom blend of MD grey and blue 600
   * See https://meyerweb.com/eric/tools/color-blend/#546E7A:1E88E5:5:hex */
  --jp-cell-inprompt-font-color: #307fc1;
  /* A custom blend of MD grey and orange 600
   * https://meyerweb.com/eric/tools/color-blend/#546E7A:F4511E:5:hex */
  --jp-cell-outprompt-font-color: #bf5b3d;

  /* Notebook specific styles */

  --jp-notebook-padding: 10px;
  --jp-notebook-select-background: var(--jp-layout-color1);
  --jp-notebook-multiselected-color: var(--md-blue-50);

  /* The scroll padding is calculated to fill enough space at the bottom of the
  notebook to show one single-line cell (with appropriate padding) at the top
  when the notebook is scrolled all the way to the bottom. We also subtract one
  pixel so that no scrollbar appears if we have just one single-line cell in the
  notebook. This padding is to enable a 'scroll past end' feature in a notebook.
  */
  --jp-notebook-scroll-padding: calc(
    100% - var(--jp-code-font-size) * var(--jp-code-line-height) -
      var(--jp-code-padding) - var(--jp-cell-padding) - 1px
  );

  /* Rendermime styles */

  --jp-rendermime-error-background: #fdd;
  --jp-rendermime-table-row-background: var(--md-grey-100);
  --jp-rendermime-table-row-hover-background: var(--md-light-blue-50);

  /* Dialog specific styles */

  --jp-dialog-background: rgba(0, 0, 0, 0.25);

  /* Console specific styles */

  --jp-console-padding: 10px;

  /* Toolbar specific styles */

  --jp-toolbar-border-color: var(--jp-border-color1);
  --jp-toolbar-micro-height: 8px;
  --jp-toolbar-background: var(--jp-layout-color1);
  --jp-toolbar-box-shadow: 0px 0px 2px 0px rgba(0, 0, 0, 0.24);
  --jp-toolbar-header-margin: 4px 4px 0px 4px;
  --jp-toolbar-active-background: var(--md-grey-300);

  /* Input field styles */

  --jp-input-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-input-active-background: var(--jp-layout-color1);
  --jp-input-hover-background: var(--jp-layout-color1);
  --jp-input-background: var(--md-grey-100);
  --jp-input-border-color: var(--jp-border-color1);
  --jp-input-active-border-color: var(--jp-brand-color1);
  --jp-input-active-box-shadow-color: rgba(19, 124, 189, 0.3);

  /* General editor styles */

  --jp-editor-selected-background: #d9d9d9;
  --jp-editor-selected-focused-background: #d7d4f0;
  --jp-editor-cursor-color: var(--jp-ui-font-color0);

  /* Code mirror specific styles */

  --jp-mirror-editor-keyword-color: #008000;
  --jp-mirror-editor-atom-color: #88f;
  --jp-mirror-editor-number-color: #080;
  --jp-mirror-editor-def-color: #00f;
  --jp-mirror-editor-variable-color: var(--md-grey-900);
  --jp-mirror-editor-variable-2-color: #05a;
  --jp-mirror-editor-variable-3-color: #085;
  --jp-mirror-editor-punctuation-color: #05a;
  --jp-mirror-editor-property-color: #05a;
  --jp-mirror-editor-operator-color: #aa22ff;
  --jp-mirror-editor-comment-color: #408080;
  --jp-mirror-editor-string-color: #ba2121;
  --jp-mirror-editor-string-2-color: #708;
  --jp-mirror-editor-meta-color: #aa22ff;
  --jp-mirror-editor-qualifier-color: #555;
  --jp-mirror-editor-builtin-color: #008000;
  --jp-mirror-editor-bracket-color: #997;
  --jp-mirror-editor-tag-color: #170;
  --jp-mirror-editor-attribute-color: #00c;
  --jp-mirror-editor-header-color: blue;
  --jp-mirror-editor-quote-color: #090;
  --jp-mirror-editor-link-color: #00c;
  --jp-mirror-editor-error-color: #f00;
  --jp-mirror-editor-hr-color: #999;

  /* Vega extension styles */

  --jp-vega-background: white;

  /* Sidebar-related styles */

  --jp-sidebar-min-width: 250px;

  /* Search-related styles */

  --jp-search-toggle-off-opacity: 0.5;
  --jp-search-toggle-hover-opacity: 0.8;
  --jp-search-toggle-on-opacity: 1;
  --jp-search-selected-match-background-color: rgb(245, 200, 0);
  --jp-search-selected-match-color: black;
  --jp-search-unselected-match-background-color: var(
    --jp-inverse-layout-color0
  );
  --jp-search-unselected-match-color: var(--jp-ui-inverse-font-color0);

  /* Icon colors that work well with light or dark backgrounds */
  --jp-icon-contrast-color0: var(--md-purple-600);
  --jp-icon-contrast-color1: var(--md-green-600);
  --jp-icon-contrast-color2: var(--md-pink-600);
  --jp-icon-contrast-color3: var(--md-blue-600);
}
</style>

<style type="text/css">
a.anchor-link {
   display: none;
}
.highlight  {
    margin: 0.4em;
}

/* Input area styling */
.jp-InputArea {
    overflow: hidden;
}

.jp-InputArea-editor {
    overflow: hidden;
}

@media print {
  body {
    margin: 0;
  }
}
</style>

<!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-MML-AM_CHTML-full,Safe"> </script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    init_mathjax = function() {
        if (window.MathJax) {
        // MathJax loaded
            MathJax.Hub.Config({
                TeX: {
                    equationNumbers: {
                    autoNumber: "AMS",
                    useLabelIds: true
                    }
                },
                tex2jax: {
                    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
                    processEscapes: true,
                    processEnvironments: true
                },
                displayAlign: 'center',
                CommonHTML: {
                    linebreaks: { 
                    automatic: true 
                    }
                },
                "HTML-CSS": {
                    linebreaks: { 
                    automatic: true 
                    }
                }
            });
        
            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    }
    init_mathjax();
    </script>
    <!-- End of mathjax configuration --></head>
<body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">

<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h1 id="Data-Mining-Project">Data Mining Project<a class="anchor-link" href="#Data-Mining-Project">&#182;</a></h1><p>Developed by Gil Teixeira, 88194.</p>
<p>You might already be on my website but if you're not feel free to come by and test the outputs of this models:</p>
<p><a href="https://ed.bearkillerpt.xyz/">https://ed.bearkillerpt.xyz/</a></p>
<p>The paper built a website just with flask.
I'm personally very fond of js and particularly ReactJs and so I used Flask to build an api that I can query to get the results!
If you wish to use it an example of a request, as available on the dev console of your browser:</p>
<p><a href="https://edapi.bearkillerpt.xyz/?age=20&amp;gender=0&amp;hypertension=0&amp;heart_disease=0&amp;ever_married=0&amp;work_type=0&amp;Residence_type=20&amp;avg_glucose_level=20&amp;bmi=20&amp;smoking_status=0">https://edapi.bearkillerpt.xyz/?age=20&amp;gender=0&amp;hypertension=0&amp;heart_disease=0&amp;ever_married=0&amp;work_type=0&amp;Residence_type=20&amp;avg_glucose_level=20&amp;bmi=20&amp;smoking_status=0</a>
For more information on how the parameters were encoded check the web-api.py script on my github page:</p>
<p><a href="https://github.com/bearkillerPT/DataMining/blob/main/WebApp/web-api.py">https://github.com/bearkillerPT/DataMining/blob/main/WebApp/web-api.py</a></p>
<p>The paper selected is named:</p>
<h3 id="Analyzing-the-Performance-of-Stroke-Prediction-using-ML-Classification-Algorithms-%5B1%5D">Analyzing the Performance of Stroke Prediction using ML Classification Algorithms [1]<a class="anchor-link" href="#Analyzing-the-Performance-of-Stroke-Prediction-using-ML-Classification-Algorithms-%5B1%5D">&#182;</a></h3>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">sklearn</span> <span class="k">as</span> <span class="nn">sk</span>
<span class="kn">import</span> <span class="nn">imblearn</span>
<span class="kn">from</span> <span class="nn">imblearn.over_sampling</span> <span class="kn">import</span> <span class="n">SMOTE</span>
<span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="kn">import</span> <span class="n">LogisticRegression</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">tree</span>
<span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="kn">import</span> <span class="n">RandomForestClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.neighbors</span> <span class="kn">import</span> <span class="n">KNeighborsClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.svm</span> <span class="kn">import</span> <span class="n">SVC</span>
<span class="kn">from</span> <span class="nn">sklearn.naive_bayes</span> <span class="kn">import</span> <span class="n">GaussianNB</span>

<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">train_test_split</span>
<span class="kn">import</span> <span class="nn">sklearn.metrics</span> <span class="k">as</span> <span class="nn">metrics</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Using pandas </span><span class="si">%s</span><span class="s2"> version&quot;</span> <span class="o">%</span> <span class="n">pd</span><span class="o">.</span><span class="n">__version__</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Using numpy </span><span class="si">%s</span><span class="s2"> version&quot;</span> <span class="o">%</span> <span class="n">np</span><span class="o">.</span><span class="n">__version__</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Using sklearn </span><span class="si">%s</span><span class="s2"> version&quot;</span> <span class="o">%</span> <span class="n">sk</span><span class="o">.</span><span class="n">__version__</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Using imblearn </span><span class="si">%s</span><span class="s2"> version&quot;</span> <span class="o">%</span> <span class="n">imblearn</span><span class="o">.</span><span class="n">__version__</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Using pandas 1.3.4 version
Using numpy 1.20.3 version
Using sklearn 1.0.2 version
Using imblearn 0.9.0 version
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Index">Index<a class="anchor-link" href="#Index">&#182;</a></h2><ul>
<li>1 - <a href="#11">Pre-processing</a><ul>
<li>1.1 - <a href="#11">Cleaning and encoding</a></li>
<li>1.2 - <a href="#12">Handling imbalanced data</a></li>
</ul>
</li>
<li>2 - <a href="#21">Machine Learning Algorithm Models</a><ul>
<li>2.1 - <a href="#21">Logistic Regression</a></li>
<li>2.2 - <a href="#22">Decision Tree</a></li>
<li>2.3 - <a href="#23">Random Fores</a></li>
<li>2.4 - <a href="#24">K-Nearest-Neighbors</a></li>
<li>2.5 - <a href="#25">SVM</a></li>
<li>2.6 - <a href="#26">Naive Bayes</a></li>
</ul>
</li>
<li>3 - <a href="#3">Results</a></li>
<li>4 - <a href="#4">Conclusion</a></li>
<li>5 - <a href="#5">References</a></li>
</ul>
<p><a name="11"></a>
First the data must be PreProcessed!
We drop the id column as it doesn't make any diference for the models performance.
Then we must substitute NaN (null) entries with the class means in this case, as pointed by the paper, exclusively the column bmi.
Finally we must encode the 5 non-numerical variables to numeric ones!</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dataset</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;./dataset/healthcare-dataset-stroke-data.csv&#39;</span><span class="p">)</span>
<span class="n">df_dataset</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">data</span><span class="o">=</span><span class="n">dataset</span><span class="p">)</span>

<span class="c1">#Dropping &#39;id&#39; column</span>
<span class="n">df_dataset</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s1">&#39;id&#39;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="c1">#Substituting NaN (null) entries with the bmi means</span>
<span class="n">df_dataset</span><span class="p">[</span><span class="s1">&#39;bmi&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">df_dataset</span><span class="p">[</span><span class="s1">&#39;bmi&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">mean</span><span class="p">(),</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">categoric_vars</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;gender&quot;</span><span class="p">,</span><span class="s2">&quot;ever_married&quot;</span><span class="p">,</span><span class="s2">&quot;work_type&quot;</span><span class="p">,</span><span class="s2">&quot;Residence_type&quot;</span><span class="p">,</span><span class="s2">&quot;smoking_status&quot;</span><span class="p">]</span>
<span class="c1">#Encoding of non-numerical variables!</span>
<span class="k">for</span> <span class="n">categoric_var</span> <span class="ow">in</span> <span class="n">categoric_vars</span><span class="p">:</span>
    <span class="n">df_dataset</span><span class="p">[</span><span class="n">categoric_var</span><span class="p">]</span><span class="o">.</span><span class="n">replace</span><span class="p">({</span><span class="n">label</span><span class="p">:</span> <span class="nb">int</span><span class="p">(</span><span class="n">idx</span><span class="p">)</span> <span class="k">for</span> <span class="n">idx</span><span class="p">,</span> <span class="n">label</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">unique</span><span class="p">(</span><span class="n">df_dataset</span><span class="p">[</span><span class="n">categoric_var</span><span class="p">]))},</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="n">df_dataset</span><span class="o">.</span><span class="n">head</span><span class="p">())</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>   gender   age  hypertension  heart_disease  ever_married  work_type  \
0       1  67.0             0              1             1          2   
1       0  61.0             0              0             1          3   
2       1  80.0             0              1             1          2   
3       0  49.0             0              0             1          2   
4       0  79.0             1              0             1          3   

   Residence_type  avg_glucose_level        bmi  smoking_status  stroke  
0               1             228.69  36.600000               1       1  
1               0             202.21  28.893237               2       1  
2               0             105.92  32.500000               2       1  
3               1             171.23  34.400000               3       1  
4               0             174.12  24.000000               2       1  
</pre>
</div>
</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p><a name="12"></a></p>
<h3 id="The-big-imbalance">The big imbalance<a class="anchor-link" href="#The-big-imbalance">&#182;</a></h3><p>As noted by the authors the data is highly imbalanced!
As shown in the plot bellow there are way more entries for no risk of stroke then for having risk of one. Of the 5110 entries only 249 correspond to patients with risk of stroke!
The paper suggest to under-sample the 'no rist' to match the amount of 'risk' entries. 
In this notebook we will also expolore over-sampling the 'risk' entries to match the 'no risk' ones! This might be advantageous as more information is passed about the data set while not having a huge bias twoards 'no risk'.</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">plot_imbalance</span><span class="p">(</span><span class="n">dataset</span><span class="p">):</span>
        <span class="n">no_risk_count</span> <span class="o">=</span> <span class="n">dataset</span><span class="p">[</span><span class="s2">&quot;stroke&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">dataset</span><span class="p">[</span><span class="s2">&quot;stroke&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">count</span><span class="p">()</span>
        <span class="n">risk_count</span> <span class="o">=</span> <span class="n">dataset</span><span class="p">[</span><span class="s2">&quot;stroke&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">dataset</span><span class="p">[</span><span class="s2">&quot;stroke&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">count</span><span class="p">()</span>
        <span class="n">bar_data</span> <span class="o">=</span> <span class="p">[</span><span class="n">no_risk_count</span><span class="p">,</span> <span class="n">risk_count</span><span class="p">]</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">4</span><span class="p">,</span><span class="mi">4</span><span class="p">))</span>
        <span class="n">bar_plot</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">bar</span><span class="p">([</span><span class="s2">&quot;No risk&quot;</span><span class="p">,</span> <span class="s2">&quot;Risk&quot;</span><span class="p">],</span> <span class="n">bar_data</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;#AE76A6&#39;</span><span class="p">,</span> <span class="s1">&#39;#993955&#39;</span><span class="p">])</span>
        <span class="k">for</span> <span class="n">idx</span><span class="p">,</span><span class="n">rect</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">bar_plot</span><span class="p">):</span>
                <span class="n">height</span> <span class="o">=</span> <span class="n">rect</span><span class="o">.</span><span class="n">get_height</span><span class="p">()</span>
                <span class="n">plt</span><span class="o">.</span><span class="n">text</span><span class="p">(</span><span class="n">rect</span><span class="o">.</span><span class="n">get_x</span><span class="p">()</span> <span class="o">+</span> <span class="n">rect</span><span class="o">.</span><span class="n">get_width</span><span class="p">()</span><span class="o">/</span><span class="mf">2.</span><span class="p">,</span> <span class="mf">0.9</span><span class="o">*</span><span class="n">height</span><span class="p">,</span>
                        <span class="n">bar_data</span><span class="p">[</span><span class="n">idx</span><span class="p">],</span>
                        <span class="n">ha</span><span class="o">=</span><span class="s1">&#39;center&#39;</span><span class="p">,</span> <span class="n">va</span><span class="o">=</span><span class="s1">&#39;bottom&#39;</span><span class="p">,</span> <span class="n">rotation</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;The high imbalance&quot;</span><span class="p">)</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">&quot;Number of entries&quot;</span><span class="p">)</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
<span class="n">plot_imbalance</span><span class="p">(</span><span class="n">df_dataset</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>To handle this imbalance and as not to produce models with very high accuracy but with very poor results in other very important metrics that will be used to compare the multiple models addressed!
The technique used in the paper references '7 Techniques to Handle Imbalanced Data – Kdnuggets' and the majority class is under-sampled to match the minority class.
Over-sampling will also be performed to compare both!</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#Over-sampling</span>
<span class="n">oversample</span> <span class="o">=</span> <span class="n">SMOTE</span><span class="p">()</span>
<span class="n">over_x</span><span class="p">,</span> <span class="n">over_y</span> <span class="o">=</span> <span class="n">oversample</span><span class="o">.</span><span class="n">fit_resample</span><span class="p">(</span><span class="n">df_dataset</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span>
    <span class="s1">&#39;stroke&#39;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">),</span> <span class="n">df_dataset</span><span class="p">[</span><span class="s1">&#39;stroke&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">&#39;int&#39;</span><span class="p">)</span> <span class="p">)</span>
<span class="n">oversampled_data</span> <span class="o">=</span> <span class="n">over_x</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">over_y</span><span class="p">)</span>

<span class="c1">#Under-sampling</span>
<span class="n">df_dataset</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_dataset</span><span class="p">[</span><span class="s2">&quot;stroke&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_dataset</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_dataset</span><span class="p">[</span><span class="s2">&quot;stroke&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">sample</span><span class="p">(</span><span class="mi">249</span><span class="p">)</span>
<span class="n">df_dataset</span><span class="o">.</span><span class="n">dropna</span><span class="p">(</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">df_dataset</span><span class="o">.</span><span class="n">reset_index</span><span class="p">(</span><span class="n">drop</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>


<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;After Under-sample:&quot;</span><span class="p">)</span>
<span class="n">plot_imbalance</span><span class="p">(</span><span class="n">df_dataset</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;After Over-sample:&quot;</span><span class="p">)</span>
<span class="n">plot_imbalance</span><span class="p">(</span><span class="n">oversampled_data</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>After Under-sample:
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>After Over-sample:
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Since our prediction target is boolean it makes a lot of sense to try first the logistic regression!
The results for the multiple metrics throughout this notebook will change per each run since the train and test sets have different samples!
<a name="21"></a></p>
<h2 id="Machine-Learning-Algorithm-Models">Machine Learning Algorithm Models<a class="anchor-link" href="#Machine-Learning-Algorithm-Models">&#182;</a></h2><p>Before modeling we must separate our data intro training and testing data sets!
Furthermore some useful functions that will be reused by every model are defined here.
Moreover it's important to note that from now we will build every model for both uner and over-sampled data!</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">x_train</span><span class="p">,</span> <span class="n">x_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">df_dataset</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span>
    <span class="s1">&#39;stroke&#39;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">),</span> <span class="n">df_dataset</span><span class="p">[</span><span class="s1">&#39;stroke&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">&#39;int&#39;</span><span class="p">),</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">)</span>
<span class="n">over_x_train</span><span class="p">,</span> <span class="n">over_x_test</span><span class="p">,</span> <span class="n">over_y_train</span><span class="p">,</span> <span class="n">over_y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">over_x</span><span class="p">,</span> <span class="n">over_y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">)</span>
<span class="n">stats</span> <span class="o">=</span> <span class="p">{}</span>
<span class="c1"># useful functions for the model stats!</span>


<span class="k">def</span> <span class="nf">print_stats</span><span class="p">(</span><span class="n">cls_name</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">prediction</span><span class="p">):</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">cls_name</span> <span class="o">+</span> <span class="s2">&quot; metrics:&quot;</span><span class="p">)</span>
    <span class="n">sample_type</span> <span class="o">=</span> <span class="n">cls_name</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s1">&#39; &#39;</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">cls_name</span> <span class="o">=</span> <span class="s1">&#39; &#39;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">cls_name</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s1">&#39; &#39;</span><span class="p">)[</span><span class="mi">1</span><span class="p">:])</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">sample_type</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">cls_name</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">cls_name</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">stats</span><span class="o">.</span><span class="n">keys</span><span class="p">():</span>
        <span class="n">stats</span><span class="p">[</span><span class="n">cls_name</span><span class="p">]</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="n">stats</span><span class="p">[</span><span class="n">cls_name</span><span class="p">][</span><span class="n">sample_type</span><span class="p">]</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="n">stats</span><span class="p">[</span><span class="n">cls_name</span><span class="p">][</span><span class="n">sample_type</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;accuracy&quot;</span><span class="p">:</span>  <span class="n">metrics</span><span class="o">.</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">prediction</span><span class="p">),</span>
        <span class="s2">&quot;precision&quot;</span><span class="p">:</span>  <span class="n">metrics</span><span class="o">.</span><span class="n">average_precision_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">prediction</span><span class="p">),</span>
        <span class="s2">&quot;recall&quot;</span><span class="p">:</span>  <span class="n">metrics</span><span class="o">.</span><span class="n">recall_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">prediction</span><span class="p">),</span>
        <span class="s2">&quot;f1_score&quot;</span><span class="p">:</span> <span class="n">metrics</span><span class="o">.</span><span class="n">f1_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">prediction</span><span class="p">)</span>
    <span class="p">}</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Accuracy: &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">stats</span><span class="p">[</span><span class="n">cls_name</span><span class="p">][</span><span class="n">sample_type</span><span class="p">][</span><span class="s2">&quot;accuracy&quot;</span><span class="p">]))</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Precision: &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">stats</span><span class="p">[</span><span class="n">cls_name</span><span class="p">][</span><span class="n">sample_type</span><span class="p">][</span><span class="s2">&quot;precision&quot;</span><span class="p">]))</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Recall: &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">stats</span><span class="p">[</span><span class="n">cls_name</span><span class="p">][</span><span class="n">sample_type</span><span class="p">][</span><span class="s2">&quot;recall&quot;</span><span class="p">]))</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;F1 score: &quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">stats</span><span class="p">[</span><span class="n">cls_name</span><span class="p">][</span><span class="n">sample_type</span><span class="p">][</span><span class="s2">&quot;f1_score&quot;</span><span class="p">]))</span>


<span class="k">def</span> <span class="nf">show_roc_curve</span><span class="p">(</span><span class="bp">cls</span><span class="p">,</span> <span class="n">x_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">):</span>
    <span class="n">probs</span> <span class="o">=</span> <span class="bp">cls</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">x_test</span><span class="p">)</span>
    <span class="n">preds</span> <span class="o">=</span> <span class="n">probs</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">]</span>
    <span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">,</span> <span class="n">threshold</span> <span class="o">=</span> <span class="n">metrics</span><span class="o">.</span><span class="n">roc_curve</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">preds</span><span class="p">)</span>
    <span class="n">roc_auc</span> <span class="o">=</span> <span class="n">metrics</span><span class="o">.</span><span class="n">auc</span><span class="p">(</span><span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Receiver operating characteristic (ROC) curve:&quot;</span><span class="p">)</span>
    <span class="n">metrics</span><span class="o">.</span><span class="n">RocCurveDisplay</span><span class="p">(</span><span class="n">fpr</span><span class="o">=</span><span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="o">=</span><span class="n">tpr</span><span class="p">,</span> <span class="n">roc_auc</span><span class="o">=</span><span class="n">roc_auc</span><span class="p">)</span><span class="o">.</span><span class="n">plot</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="Logistic-Regression">Logistic Regression<a class="anchor-link" href="#Logistic-Regression">&#182;</a></h3><h4 id="Under-sampling">Under-sampling<a class="anchor-link" href="#Under-sampling">&#182;</a></h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">logistic_classifier</span> <span class="o">=</span> <span class="n">LogisticRegression</span><span class="p">(</span><span class="n">max_iter</span><span class="o">=</span><span class="mi">1000</span><span class="p">)</span>
<span class="n">under_log_model</span> <span class="o">=</span> <span class="n">logistic_classifier</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
<span class="n">under_prediction</span> <span class="o">=</span> <span class="n">logistic_classifier</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x_test</span><span class="p">)</span>
<span class="n">print_stats</span><span class="p">(</span><span class="s2">&quot;Under-sampled Logistic Regression&quot;</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">under_prediction</span><span class="p">)</span>
<span class="n">show_roc_curve</span><span class="p">(</span><span class="n">under_log_model</span><span class="p">,</span> <span class="n">x_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Under-sampled Logistic Regression metrics:
Under-sampled
Logistic Regression
Accuracy: 0.78
Precision: 0.77
Recall: 0.75
F1 score: 0.7924528301886793
Receiver operating characteristic (ROC) curve:
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h4 id="Over-sampling">Over-sampling<a class="anchor-link" href="#Over-sampling">&#182;</a></h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">over_log_model</span> <span class="o">=</span> <span class="n">logistic_classifier</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">over_x_train</span><span class="p">,</span> <span class="n">over_y_train</span><span class="p">)</span>
<span class="n">over_prediction</span> <span class="o">=</span> <span class="n">logistic_classifier</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">over_x_test</span><span class="p">)</span>
<span class="n">print_stats</span><span class="p">(</span><span class="s2">&quot;Over-sampled Logistic Regression&quot;</span><span class="p">,</span> <span class="n">over_y_test</span><span class="p">,</span> <span class="n">over_prediction</span><span class="p">)</span>
<span class="n">show_roc_curve</span><span class="p">(</span><span class="n">over_log_model</span><span class="p">,</span> <span class="n">over_x_test</span><span class="p">,</span> <span class="n">over_y_test</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Over-sampled Logistic Regression metrics:
Over-sampled
Logistic Regression
Accuracy: 0.8051413881748072
Precision: 0.7425139429046461
Recall: 0.819773429454171
F1 score: 0.8077118214104516
Receiver operating characteristic (ROC) curve:
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p><a name="22"></a></p>
<h3 id="Decision-Tree-Classification">Decision Tree Classification<a class="anchor-link" href="#Decision-Tree-Classification">&#182;</a></h3><h4 id="Under-sampling">Under-sampling<a class="anchor-link" href="#Under-sampling">&#182;</a></h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">decision_tree_cls</span> <span class="o">=</span> <span class="n">tree</span><span class="o">.</span><span class="n">DecisionTreeClassifier</span><span class="p">()</span>
<span class="n">under_decision_tree_model</span> <span class="o">=</span> <span class="n">decision_tree_cls</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
<span class="n">prediction</span> <span class="o">=</span> <span class="n">under_decision_tree_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x_test</span><span class="p">)</span>
<span class="n">print_stats</span><span class="p">(</span><span class="s2">&quot;Under-sampled Decision Tree&quot;</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">prediction</span><span class="p">)</span>
<span class="n">show_roc_curve</span><span class="p">(</span><span class="n">under_decision_tree_model</span><span class="p">,</span> <span class="n">x_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Under-sampled Decision Tree metrics:
Under-sampled
Decision Tree
Accuracy: 0.74
Precision: 0.7233333333333334
Recall: 0.75
F1 score: 0.7636363636363638
Receiver operating characteristic (ROC) curve:
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h4 id="Over-sampling">Over-sampling<a class="anchor-link" href="#Over-sampling">&#182;</a></h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">over_decision_tree_model</span> <span class="o">=</span> <span class="n">decision_tree_cls</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">over_x_train</span><span class="p">,</span> <span class="n">over_y_train</span><span class="p">)</span>
<span class="n">prediction</span> <span class="o">=</span> <span class="n">under_decision_tree_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">over_x_test</span><span class="p">)</span>
<span class="n">print_stats</span><span class="p">(</span><span class="s2">&quot;Over-sampled Decision Tree&quot;</span><span class="p">,</span> <span class="n">over_y_test</span><span class="p">,</span> <span class="n">prediction</span><span class="p">)</span>
<span class="n">show_roc_curve</span><span class="p">(</span><span class="n">over_decision_tree_model</span><span class="p">,</span> <span class="n">over_x_test</span><span class="p">,</span> <span class="n">over_y_test</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Over-sampled Decision Tree metrics:
Over-sampled
Decision Tree
Accuracy: 0.9177377892030848
Precision: 0.8782635891883772
Recall: 0.9320288362512873
F1 score: 0.9187817258883249
Receiver operating characteristic (ROC) curve:
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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iXzgAx/gzTffBOCnP/0py5cv58QTT2T58uU88sgjE44HEpYIBkcMNU0rciQiUkhr1qzhne98J2vXro38mc997nPs2LGDTZs2sWnTJu677z727t07oTi2bNnC2rVr2bx5Mw899BBXXnklfX19B5T72Mc+xvXXX8/GjRu58MILueGGGwBobGzkvvvuY+PGjXznO9/h0ksvnVA8AxI1aqhlYLI53RGIFNw/3beZLa+/mddjLp09nX/8wPGjlslkMjz++OM8+uijnH/++XzhC18Y87gdHR1861vf4uWXX6a2thaAww8/nEsuuWRC8f7oRz9i5cqV1NbWsnDhQhYvXsyTTz7JGWecMaTciy++yJlnngnA2Wefzfvf/36uu+46TjnllMEyxx9/PJ2dnXR1dQ3GeLASd0cwZ+YU6msSlf9EEu3ee+9lxYoVHH300Rx66KE888wzY34mlUoxf/58pk+fPmbZT3/605x88skH/Fx//fUHlN2+fTvz5s0b3J47dy7bt28/oNwJJ5zAunXrAPjhD3/Ia6+9dkCZu+66i1NOOWXCSQASdkeQCiebE5HCG+vKPS5r1qzhU5/6FAArV65kzZo1nHrqqSOOrhnvqJuvf/3rkcu6e6Tvu+222/jkJz/Jtddey/nnn09NzdC11Tdv3szVV1/NT37yk3HFOpJYE4GZrQC+AVQCt7r79TnvW/j+uUAHcJm7j52uD0J/v9OSbuetpx0ax+FFZBLauXMnjzzyCJs2bcLM6Ovrw8z4yle+wqxZs9i9e/eQ8rt27aKxsZHFixezbds29u7dy7Rpo/cpfvrTn+bRRx89YP/KlSu55pprhuybO3fukKv71tZWZs+efcBnjz322MFK/qWXXuLHP/7xkM9ceOGF3HHHHSxatGjsP0IU7h7LD0Hl3wIcBdQAzwFLc8qcCzwIGHA68Kuxjrt8+XI/GK/tavcjr77fv/fEKwf1eREZvy1bthT1+2+55RZftWrVkH1nnnmmr1+/3js7O33BggWDMb7yyis+f/58//3vf+/u7p/5zGf8sssu866uLnd3f/311/273/3uhOLZtGmTL1u2zDs7O33r1q2+cOFC7+3tPaDc7373O3d37+vr80svvdS//e1vu7v77t27fdmyZX7nnXeO+j3D/d2BDT5CvRpnH8FpQMrdt7p7N7AWuCCnzAXAHWGcTwAzzeyI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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p><a name="23"></a></p>
<h3 id="Random-Forest-Classification">Random Forest Classification<a class="anchor-link" href="#Random-Forest-Classification">&#182;</a></h3><h4 id="Under-sampling">Under-sampling<a class="anchor-link" href="#Under-sampling">&#182;</a></h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">random_forest_cls</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">()</span>
<span class="n">under_random_forest_model</span> <span class="o">=</span> <span class="n">random_forest_cls</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
<span class="n">prediction</span> <span class="o">=</span> <span class="n">under_random_forest_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x_test</span><span class="p">)</span>
<span class="n">print_stats</span><span class="p">(</span><span class="s2">&quot;Under-sampled Random Forest&quot;</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">prediction</span><span class="p">)</span>
<span class="n">show_roc_curve</span><span class="p">(</span><span class="n">under_random_forest_model</span><span class="p">,</span> <span class="n">x_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Under-sampled Random Forest metrics:
Under-sampled
Random Forest
Accuracy: 0.78
Precision: 0.7602116402116401
Recall: 0.7857142857142857
F1 score: 0.7999999999999999
Receiver operating characteristic (ROC) curve:
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h4 id="Over-sampling">Over-sampling<a class="anchor-link" href="#Over-sampling">&#182;</a></h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">over_random_forest_model</span> <span class="o">=</span> <span class="n">random_forest_cls</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">over_x_train</span><span class="p">,</span> <span class="n">over_y_train</span><span class="p">)</span>
<span class="n">prediction</span> <span class="o">=</span> <span class="n">over_random_forest_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">over_x_test</span><span class="p">)</span>
<span class="n">print_stats</span><span class="p">(</span><span class="s2">&quot;Over-sampled Random Forest&quot;</span><span class="p">,</span> <span class="n">over_y_test</span><span class="p">,</span> <span class="n">prediction</span><span class="p">)</span>
<span class="n">show_roc_curve</span><span class="p">(</span><span class="n">over_random_forest_model</span><span class="p">,</span> <span class="n">over_x_test</span><span class="p">,</span> <span class="n">over_y_test</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Over-sampled Random Forest metrics:
Over-sampled
Random Forest
Accuracy: 0.9470437017994858
Precision: 0.9155163856289102
Recall: 0.9670442842430484
F1 score: 0.9480060575466935
Receiver operating characteristic (ROC) curve:
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Like the Decision Tree model the Random Forest one behaves way better with over-sampling! 
<a name="24"></a></p>
<h3 id="K-nearest-neighbors">K-nearest neighbors<a class="anchor-link" href="#K-nearest-neighbors">&#182;</a></h3><h4 id="Under-sampling">Under-sampling<a class="anchor-link" href="#Under-sampling">&#182;</a></h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">k_nearest_neighbors_cls</span> <span class="o">=</span> <span class="n">KNeighborsClassifier</span><span class="p">()</span>
<span class="n">under_k_nearest_neighbors_model</span> <span class="o">=</span> <span class="n">k_nearest_neighbors_cls</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
<span class="n">prediction</span> <span class="o">=</span> <span class="n">under_k_nearest_neighbors_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x_test</span><span class="p">)</span>
<span class="n">print_stats</span><span class="p">(</span><span class="s2">&quot;Under-sampled K-nearest neighbors&quot;</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">prediction</span><span class="p">)</span>
<span class="n">show_roc_curve</span><span class="p">(</span><span class="n">under_k_nearest_neighbors_model</span><span class="p">,</span> <span class="n">x_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Under-sampled K-nearest neighbors metrics:
Under-sampled
K-nearest neighbors
Accuracy: 0.78
Precision: 0.7514778325123153
Recall: 0.8214285714285714
F1 score: 0.8070175438596492
Receiver operating characteristic (ROC) curve:
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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iMReRUUFhYWFlJaWpjqUtJGTk0P//v3Jykr+QlaJILTsoy389InFHDUkj28de0CqwxFpEwoLC+nSpQuDBg1SM2sLcHeKi4spLCxk8ODk50rRPQKCoaWvmL6Azu0z+cPXRtNOQ0uLNIvS0lJ69OihJNBCzIwePXo0ugamGgFw05NLWPZRCfdfdAS9umhoaZHmpCTQsvbm9532NYJ/L1rH9DmruezYAzhmWM9UhyMi0uLSOhGs2biN6x9ZxJgB3fj+54elOhwRicijjz6KmfHOO+/sXPfyyy9z2mmn7VZu0qRJPPzww0Bwo/v6669n6NChHHzwwRxxxBE8/fTTTY7lV7/6FUOGDGH48OE8++yztZZZuHAh48ePZ8yYMeTn5zN37lwA5s6dy5gxYxgzZgyjR4/m0UcfbXI8kMaJoKKqmitnvAnAn845lKyMtP1ViLR5M2bM4KijjmLmzJlJf+aGG25g3bp1FBQUUFBQwBNPPMGWLVuaFMeSJUuYOXMmixcv5plnnuHyyy+nqqpqj3LXXnstN954IwsXLuSmm27i2muvBeDggw9m/vz5LFy4kGeeeYZvfvObVFZWNikmSON7BL//zzIWrvmUO84dy4DuHVMdjkib97MnFrPkw83Nus+R+3blxi/WPyBkSUkJr732Gi+99BKnn346P/3pTxvc77Zt25g6dSorV66kffv2APTu3Zuzzz67SfH+61//YuLEibRv357BgwczZMgQ5s6dy2c+85ndypkZmzcHv6tNmzax7777AtCx465zVWlpabPdf0nLRDBr2QbueuV9zh03kC+M6pvqcEQkQo899hgTJkxg2LBhdO/enQULFjB27Nh6P7N8+XIGDhxI164NjzP23e9+l5deemmP9RMnTuT666/fbd3atWsZP378zuX+/fuzdu3aPT77xz/+kZNPPpkf/OAHVFdX8/rrr+/cNmfOHC666CJWrVrFAw88QGZm00/jaZcIPt5SyvceWsjw3l34yWkjUx2OSNpo6Mo9KjNmzODqq68GgpPzjBkzGDt2bJ1X0429yr7llluSLuvuSX3fn//8Z2655RbOPPNMHnroIS6++GKef/55AMaNG8fixYtZunQp3/jGNzjllFPIyWna046RJgIzmwDcCmQAd7v7zTW2W7j9VGAbMMndF0QVT3W1872/v0VJWSUzLh1PTpamnBRpy4qL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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h4 id="Over-Sampling">Over-Sampling<a class="anchor-link" href="#Over-Sampling">&#182;</a></h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">over_k_nearest_neighbors_model</span> <span class="o">=</span> <span class="n">k_nearest_neighbors_cls</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">over_x_train</span><span class="p">,</span> <span class="n">over_y_train</span><span class="p">)</span>
<span class="n">prediction</span> <span class="o">=</span> <span class="n">over_k_nearest_neighbors_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">over_x_test</span><span class="p">)</span>
<span class="n">print_stats</span><span class="p">(</span><span class="s2">&quot;Over-sampled K-nearest neighbors&quot;</span><span class="p">,</span> <span class="n">over_y_test</span><span class="p">,</span> <span class="n">prediction</span><span class="p">)</span>
<span class="n">show_roc_curve</span><span class="p">(</span><span class="n">over_k_nearest_neighbors_model</span><span class="p">,</span> <span class="n">over_x_test</span><span class="p">,</span> <span class="n">over_y_test</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Over-sampled K-nearest neighbors metrics:
Over-sampled
K-nearest neighbors
Accuracy: 0.8889460154241645
Precision: 0.8216611155848826
Recall: 0.9824922760041195
F1 score: 0.8983050847457628
Receiver operating characteristic (ROC) curve:
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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lWnv841vfIM9e/awadMmNm3axGOPPcaRI0c6FMeWLVtYuXIlmzdv5qmnnuLmm2+mrq7ulHI33XQTd955Jxs3buSaa67hrrvuatw2fvx4NmzYwIYNG1iyZEmH4mkQq6eGGoafVpulSOb9w2Ob2bL7cKcec+qoAfz9J85utczRo0d5/vnneeaZZ7jyyiv51re+1eZxKysr+fGPf8zOnTspKCgAYPjw4Vx//fUdivdXv/oV8+bNo6CggHHjxjFhwgReeuklLrrooveV27ZtG3PmzAHgox/9KB//+Me54447OvTZrYlXjUDDT4vEzqOPPsrcuXOZNGkSgwcP5uWXX25znx07dlBaWsqAAQPaLPvlL3+5sakm9XXnnXeeUra8vJwxY8Y0LpeUlFBeXn5KuXPOOYdVq1YB8Itf/IJ33nmncdvOnTs577zz+PCHP8xzzz3XZnzpiFmNoJJzS4qyHYZILLX1yz0qK1as4NZbbwVg3rx5rFixgvPPP7/FloH2thh8//vfT7usu6f1ecuWLeOLX/wit99+O1deeSX5+UFz9siRI9m1axdDhgxh/fr1XH311WzevDmthNWaSBOBmc0FfgDkAfe6+51Ntlu4/XKgEpjv7m2n69NwrKaWZOUJDT8tEiMHDx7k6aefZtOmTZgZdXV1mBnf/e53GTJkCMlk8n3l33vvPYYOHcqECRPYtWsXR44coX///q1+xpe//GWeeeaZU9bPmzePxYsXv29dSUnJ+37dJxIJRo0adcq+U6ZM4Te/+Q0A27dv59e//jUABQUFjU1VM2fOZPz48Wzfvp2ysrI0vo1WuHskL4KL/xvAmUA+8CowtUmZy4EnAQMuBP7Y1nFnzpzpp2Pb3sN+xm2P+6OvJE5rfxFpvy1btmT185csWeILFy5837o5c+b4mjVrvLq62seOHdsY41tvveWlpaVeUVHh7u5f/epXff78+V5TU+Pu7rt37/b777+/Q/Fs2rTJp02b5tXV1f7mm2/6uHHjvLa29pRy7777rru719XV+Q033OA/+clP3N193759jeXfeOMNHzVqlB88ePCU/Zv73oF13sJ1Ncp7BLOAHe7+prsfB1YCVzUpcxVwXxjni8BAMxsZRTDqTCYSPytWrOCaa65537prr72WBx98kIKCAn72s5+xYMECZsyYwXXXXce9995LUVHQfPztb3+b4uJi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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p><a name="25"></a></p>
<h3 id="Support-Vector-Machine">Support Vector Machine<a class="anchor-link" href="#Support-Vector-Machine">&#182;</a></h3>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">svm_cls</span> <span class="o">=</span> <span class="n">clf</span> <span class="o">=</span> <span class="n">SVC</span><span class="p">(</span><span class="n">kernel</span><span class="o">=</span><span class="s1">&#39;rbf&#39;</span><span class="p">,</span> <span class="n">probability</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">under_svm_model</span> <span class="o">=</span> <span class="n">svm_cls</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
<span class="n">prediction</span> <span class="o">=</span> <span class="n">under_svm_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x_test</span><span class="p">)</span>
<span class="n">print_stats</span><span class="p">(</span><span class="s2">&quot;Under-sampled SVM&quot;</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">prediction</span><span class="p">)</span>
<span class="n">show_roc_curve</span><span class="p">(</span><span class="n">under_svm_model</span><span class="p">,</span> <span class="n">x_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Under-sampled SVM metrics:
Under-sampled
SVM
Accuracy: 0.8
Precision: 0.7796428571428572
Recall: 0.8035714285714286
F1 score: 0.8181818181818182
Receiver operating characteristic (ROC) curve:
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h4 id="Over-sampling">Over-sampling<a class="anchor-link" href="#Over-sampling">&#182;</a></h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">over_svm_model</span> <span class="o">=</span> <span class="n">svm_cls</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">over_x_train</span><span class="p">,</span> <span class="n">over_y_train</span><span class="p">)</span>
<span class="n">prediction</span> <span class="o">=</span> <span class="n">over_svm_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">over_x_test</span><span class="p">)</span>
<span class="n">print_stats</span><span class="p">(</span><span class="s2">&quot;Over-sampled SVM&quot;</span><span class="p">,</span> <span class="n">over_y_test</span><span class="p">,</span> <span class="n">prediction</span><span class="p">)</span>
<span class="n">show_roc_curve</span><span class="p">(</span><span class="n">over_svm_model</span><span class="p">,</span> <span class="n">over_x_test</span><span class="p">,</span> <span class="n">over_y_test</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Over-sampled SVM metrics:
Over-sampled
SVM
Accuracy: 0.7696658097686375
Precision: 0.7006407637383814
Recall: 0.8177136972193615
F1 score: 0.7799607072691552
Receiver operating characteristic (ROC) curve:
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYIAAAEGCAYAAABo25JHAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAAopElEQVR4nO3deXxU5dn/8c9lEhL2XWQx7KigrBGhirViFa3rY+tW258+tZS2bn0eqzytVau2ttpN60JxqVUrtLVarUWxVhYVkU1kFWQRCKCsshrIcv3+OIcYQpYJycnJzHzfr9e8Muece858TwJzzdnu29wdERFJX0fEHUBEROKlQiAikuZUCERE0pwKgYhImlMhEBFJc5lxB6ipdu3aebdu3eKOISKSVObOnbvF3dtXtCzpCkG3bt2YM2dO3DFERJKKma2pbJkODYmIpDkVAhGRNKdCICKS5lQIRETSnAqBiEiai6wQmNkTZrbJzBZVstzM7AEzW2FmC8xscFRZRESkclHuETwJjKpi+dlA7/AxGngkwiwiIlKJyO4jcPfpZtatiiYXAE950A/2TDNrZWYd3X1jVJlEJH5FxSVs27P/kPlLNu6ksLhuusV/Z+VWmmVn1Mm6GpK8bm04tU+F94TVSpw3lHUG1pWZzg/nHVIIzGw0wV4Dubm59RJOpKHYV1TMmq17Wb/9M7Dar293QRGLN+ykSaOKPyhnrNxCs+ys2r8R8OaHm2mek0VGmWMPn+zcVyfrToTVwe+rIRnzxZ4pVwgq+hNV+HXA3ccD4wHy8vI0ko6krE07C1j2yS6WfbyLHZ8V8sjUlRSVxPNPvm/HFrVeR4/2zdi7v4jhPdqWzjswFlb/o1se1HZ/UQkndG5JTlbtv8lnZRxBnw7NsFSrBBGJsxDkA0eXme4CbIgpi0idKywuYX9RSen04g07eXfVVrKzjmDJhuAwyIHPqU/3FvLWii2Vruubw7vSv0srerZvWifZWjTOolvbyteVcYQ+QNNJnIXgJeBaM5sInATs0PkBSRb7iopZtH5H6bfbLbv38ce3P2LjjgLMoMSddds+q3Y9PcIPdndo16wRbZtm89UhXRhwdCu6tWtC6yaNyMrQVd4SrcgKgZlNAE4D2plZPnA7kAXg7uOAScA5wApgL3B1VFlE6oK7s3TjLv46Zx1Pzvio0naNszIYedyRHNOhBZ1a5dCldWMACoud4zu3JK9rawBysjL0zVsahCivGrq8muUOfD+q9xepKwWFxfzf8wt54b31B83v2b4pd5zfr3S6TdNG9OvUsvzLRRq8pOuGWqQ+bdzxGcPveaN0unOrxtx94fHkdWtN85y6ubJGJG4qBJL29hUV8+eZa7l38gcUFAYndxuFx+X3F39+snfpnaNoXMkllyLJTIVA0tI7K7fyyqKNLN24k9kfbS+dn5VhnHNCRzq1Co7ru0P75tl865TucUUViZwKgaSV5+flM2HW2tIP/8zwZO3Jvdry+8sH06ZpozjjicRChUDSQmFxCQN++hp79xcDcHznFpxzQke+d1qvmJOJxE+FQFLWzoJCZqzYyrhpK5m/7tPS+Q9eMYhz+3eKL5hIA6NCICnnrQ+3cOXj7x4078jm2Qw4uhW/vmQALXS1j8hBVAgk6c1ctZW7/7WE9ds/o0mjTNZ/+vkdvbd+5Th6HdmM0445MsaEIg2bCoEkJXfn4akrmbV6G9OWbwageU4mx3duwdDubTi5Vzu+OqRLzClFkoMKgSSNfUXFLMjfwWuLP+bRN1eXzm/VJIuLBnXm9vP6VfFqEamMCoEkhXteWcofpq06aF5WhvHaD75I93Z10yOnSLpSIZAGb/2nn5UWga8N6cKFgzozKLcVTRrpn69IXdD/JGnQHnzjQ3712nIArhyWy90XnhBzIpHUo0IgDc7UZZu46o+zD5qX26YJd11wfEyJRFKbCoE0CEXFJVz08AzWbN3DzoIiAAZ0acmxR7Xg+jN60zns+0dE6p4KgcRmy+59LMzfwb6iEsY8M7d0/pHNs7n13L6cP0B3/4rUBxUCqTevL/mEOWu2s2dfEU/PXFNhmxU/O5tMDc0oUq9UCCQyH+8oYP66TxnzzFyaZWeye1/RIW1+en4/+nVqQdPsTHod2UxFQCQGKgRS5xZv2MFXHnjroHm79xVx0aDOXDkslyFd28SUTEQqokIgdaa4xPn3kk8OOt5/5wX9GNajLX06NI8xmYhURYVA6sTdLy/hsbc+7/bhB2f04fqRvTCzGFOJSCJUCKRWSkqcEfdOKe3x8ysndOTiIZ05/dgOMScTkUSpEEitPDcvv7QIvHrjCI49qkXMiUSkplQIpFaKih2AN2/+Eke3aRJzGhE5HLpWT2pl3tpgEPhGmfqnJJKs9L9XamXl5t0AtGys4R9FkpUKgRy2v81Zx3trP6VH+6bkZGXEHUdEDpMKgRy2qcuCISJvGNk75iQiUhsqBHLYMo4wurdrygUDO8cdRURqQYVARCTNqRDIYdlXVMxL72+gxD3uKCJSS5EWAjMbZWbLzGyFmY2tYHlLM/unmb1vZovN7Ooo80jdefG9DQCoAwmR5BdZITCzDOAh4GygL3C5mfUt1+z7wBJ3HwCcBvzazBpFlUnqRmFxCTf/fQEAT149NOY0IlJbUe4RDAVWuPsqd98PTAQuKNfGgeYW9EzWDNgGHNppvTQo97/+IQB9O7agW7umMacRkdqKshB0BtaVmc4P55X1IHAcsAFYCNzg7iXlV2Rmo81sjpnN2bx5c1R5JQFTPtjE+DdXkZ15BBO/MyzuOCJSB6IsBBUdPi5/ZvEsYD7QCRgIPGhmh/Ra5u7j3T3P3fPat29f1zklQS8v2MC3n5pDnw7NmDH2dFrk6G5ikVQQZadz+cDRZaa7EHzzL+tq4Bfu7sAKM1sNHAvMijCX1NC2Pfv54d/e5z8fbGJI19b88eoTVQREUkiUhWA20NvMugPrgcuAK8q1WQuMBN40sw7AMcCqCDPJYRh8179Lnz/y9cEqAiIpJrJC4O5FZnYtMBnIAJ5w98VmNiZcPg64C3jSzBYSHEq6xd23RJVJaubR6at4csZHpdOLf3oWTbPVc7lIqon0f7W7TwImlZs3rszzDcCZUWaQmisqLuHKx99l5qptpfPe+N8vqgiIpCj9z5ZDTFu+ubQIaMAZkdSnLibkED/4y3wAXrr2ZBUBkTSgQiAHeXH+enYWBPf0ndC5ZcxpRKQ+6NCQsGNvIZeOf4ctu/exZfd+AB7++mCCG75FJNWpEKQxd+f+/3zI78IuIwC+0LMtlw3N5ZwTOsaYTETqkwpBmvlsfzHz133KvLXbuW/ystL53x7RnZvOOobsTA05KZJuVAjSxKadBQz9+X8Omd+ycRZPXn0ig3Jbx5BKRBoCFYI08c0ngl472jfP5vITjyavWxtO6NyS1k3V67dIulMhSANf/s00Pty0G4C3bzmdRpm6WExEPqdPhBQ3c9XW0iLwyg0jVARE5BAJfyqYmUYgSTLPvruWy8bPBOAP3xjCcR0P6eFbRKT6QmBmXzCzJcDScHqAmT0ceTKpld37ivjRCwsB+PlFJ3Bm3w4xJxKRhiqRPYLfEgwgsxXA3d8HTo0ylNTOjs8KOf72yQD06dCMK07K1c1hIlKphA4Nufu6crOKI8gidWD1lj0M+OlrADTOymDS9SNiTiQiDV0iVw2tM7MvAG5mjYDrCQ8TScPy+/98yK//vRyAXkc241/Xn0Jmhk4Oi0jVEikEY4D7CQaezwdeA74XZSipmYLCYiYt3FhaBL53Wk9uHnVszKlEJFkkUgiOcfevl51hZicDb0cTSWpi064Chv7s8zuGvzGsq4qAiNRIIoXg98DgBOZJPfrwk12c9+BbFBSWlM577Qen0qdD8xhTiUgyqrQQmNlw4AtAezP7nzKLWhCMQSwxWbxhB1954K3S6dvO7cvVJ3fTlUEicliq2iNoBDQL25T9mrkT+GqUoaRq89ZsB+Dqk7tx+3n9Yk4jIsmu0kLg7tOAaWb2pLuvqcdMUo2fvLgYgOtO7x1zEhFJBYmcI9hrZvcB/YCcAzPd/fTIUkmlZqzcUvq8jXoOFZE6kMhF5n8GPgC6Az8FPgJmR5hJqrB9TyEAz40ZHnMSEUkViRSCtu7+OFDo7tPc/b+BYRHnkkpMWrgRgBaNs2JOIiKpIpFDQ4Xhz41m9hVgA9AlukhSGXfnX2Eh6NFOncGKSN1IpBDcbWYtgf8luH+gBXBjlKHkUHe8tJgnZ3wEwKDcVuo6QkTqTLWFwN1fDp/uAL4EpXcWSz25/cVF/Omd4MKtEzq35In/d2LMiUQklVR1Q1kGcAlBH0OvuvsiMzsX+BHQGBhUPxFlwfodALx83Skc37llzGlEJNVUtUfwOHA0MAt4wMzWAMOBse7+j3rIJsDKzbt5b+2nnNyrrYqAiESiqkKQB/R39xIzywG2AL3c/eP6iSb7iooZ+etpABzduknMaUQkVVV1xnG/u5cAuHsBsLymRcDMRpnZMjNbYWZjK2lzmpnNN7PFZjatJutPdT/7VzDsQ4ucTH5xcf+Y04hIqqpqj+BYM1sQPjegZzhtgLt7lZ9M4TmGh4AvE4xjMNvMXnL3JWXatAIeBka5+1ozO/LwNyV1FBQWM2v1Np4KTxDP+vEZMScSkVRWVSE4rpbrHgqscPdVAGY2EbgAWFKmzRXA8+6+FsDdN9XyPZPe/qISjv3Jq6XTx3duQU6WOnsVkehU1elcbTua6wyUHes4HzipXJs+QJaZTSXo4fR+d3+q/IrMbDQwGiA3N7eWsRq2Efe+Ufr82W+fxLDubWNMIyLpIJEbyg5XRZ3jewXvPwQYSXBJ6jtmNtPdlx/0IvfxwHiAvLy88utIGT96YSGf7NwHwAd3jdKegIjUiygLQT7B5acHdCHonqJ8my3uvgfYY2bTgQHActLMhFlrefbdtQD8ZfQwFQERqTcJ9VNgZo3N7Jgarns20NvMuptZI+Ay4KVybV4ERphZppk1ITh0tLSG75MS/u/5hQA8+s08Tuqhw0EiUn+qLQRmdh4wH3g1nB5oZuU/0A/h7kXAtcBkgg/3v7r7YjMbY2ZjwjZLw/UuILhx7TF3X3SY25KUSkqcB9/4EIDzBnTiy307xJxIRNJNIoeG7iC4AmgqgLvPN7Nuiazc3ScBk8rNG1du+j7gvkTWl4qunTCPSQuD2zNG9Tsq5jQiko4SKQRF7r5DA6PXvYLC4tIi8PbY0+ncqnHMiUQkHSVSCBaZ2RVAhpn1Bq4HZkQbKz1MX74ZgEvyuqgIiEhsEjlZfB3BeMX7gGcJuqO+McJMaWHmqq2MfnouAJcPTe17I0SkYUtkj+AYd/8x8OOow6ST370eXCF74xm9GZTbOuY0IpLOEtkj+I2ZfWBmd5lZv8gTpYnG4X0CN4zsHXMSEUl31RYCd/8ScBqwGRhvZgvN7Naog6WyDz7eyZRlm+nXqQU6CS8icUvohjJ3/9jdHwDGENxTcFuUoVLdRQ8F59pz22iMARGJXyI3lB1nZneY2SLgQYIrhrpEnixFzV2znc8KizmpexsevGJw3HFERBI6WfxHYAJwpruX7ytIauCTnQVc/EiwN3Dt6b3IOEKHhUQkftUWAncfVh9B0sGfZwY9e3dokc0pvdrFnEZEJFBpITCzv7r7JWa2kIO7j05ohDI5WHGJ88AbKwCYMXakThKLSINR1R7BDeHPc+sjSKq7YeJ7AHRqmaNDQiLSoFR6stjdN4ZPv+fua8o+gO/VT7zU8Nrij3l5QfDr/Od1p8ScRkTkYIlcPvrlCuadXddBUtnjb60G4O4Lj6dts+yY04iIHKyqcwTfJfjm38PMFpRZ1Bx4O+pgqaRJowx6tm/KlcO6xh1FROQQVZ0jeBZ4BbgHGFtm/i533xZpqhTUNDvKUUFFRA5fVYeG3N0/Ar4P7CrzwMzaRB8tNewqKGTKss24V99WRCQO1e0RnAvMJbh8tOylLg70iDBXSthVUMgJd7wGQHZmQr15iIjUu0oLgbufG/7sXn9xUoe7lxYBgKe+NTTGNCIilUukr6GTzaxp+PxKM/uNmWkklWqs2LQbgMwjjJU/P4cmjXSOQEQapkSOVzwC7DWzAcDNwBrg6UhTpYC9+4sBeODyQbqBTEQatEQKQZG7O3ABcL+7309wCalUorjEueCh4ArbnCydGxCRhi2R4xW7zOz/gG8AI8wsA8iKNlZye/bdNaXPBx6tYShFpGFL5OvqpQQD1/+3u38MdAbuizRVkjtwWGjWj0bSpmmjmNOIiFQtkaEqPwb+DLQ0s3OBAnd/KvJkSWzzrn0ANMvRCWIRafgSuWroEmAW8DXgEuBdM/tq1MGS1d79RTwW9i2kk8QikgwS+cr6Y+BEd98EYGbtgdeB56IMlqy++8w8AC4a1JnszIyY04iIVC+RcwRHHCgCoa0Jvi7tFBaXMG35ZgC+80XdeC0iySGRPYJXzWwywbjFEJw8nhRdpOR305l9OPaoFnHHEBFJSCJjFv/QzP4LOIWgv6Hx7v5C5MlERKReVDUeQW/gV0BPYCFwk7uvr69gyWjy4o8BKFFPoyKSRKo61v8E8DJwMUEPpL+v6crNbJSZLTOzFWY2top2J5pZcTJfjfTvJZ9w7bPBuMTDe7aNOY2ISOKqOjTU3N0fDZ8vM7N5NVlxeAfyQwRDXeYDs83sJXdfUkG7XwKTa7L+huTxt1Zz18vBZp3bvyMndtNwDSKSPKoqBDlmNojPxyFoXHba3asrDEOBFe6+CsDMJhL0V7SkXLvrgL8DJ9Ywe4Pg7qVF4PeXD+K8AZ1iTiQiUjNVFYKNwG/KTH9cZtqB06tZd2dgXZnpfOCksg3MrDNwUbiuSguBmY0GRgPk5jasHrD/MjvYxNw2TVQERCQpVTUwzZdque6Kbqstfxr1d8At7l5sVvlduO4+HhgPkJeX16BOxd47eRkAz40ZHnMSEZHDE2VnOPnA0WWmuwAbyrXJAyaGRaAdcI6ZFbn7PyLMVWf27Cti2579ABzZIifmNCIihyfKQjAb6G1m3YH1wGXAFWUblB0G08yeBF5OliIAUFAY9DL63dN6xpxEROTwRVYI3L3IzK4luBooA3jC3Reb2Zhw+bio3ru+/HVOPgBHaW9ARJJYtYXAguM2Xwd6uPud4XjFR7n7rOpe6+6TKNcdRWUFwN2vSihxA7Ftz35++eoHAIw6/qiY04iIHL5EOo97GBgOXB5O7yK4PyCt/fKVoAh0aplDB+0RiEgSS+TQ0EnuPtjM3gNw9+1mltbDbm3fs5+/zAkuG33rluquohURadgS2SMoDO/+dSgdj6Ak0lQN3D2vLAVgSNfWHKHBZ0QkySVSCB4AXgCONLOfAW8BP480VQN3YMAZ3TsgIqkgkW6o/2xmc4GRBDeJXejuSyNP1sC1adqIqm6CExFJFolcNZQL7AX+WXaeu6+NMlhDtm3vfoqK0/romIikkEROFv+L4PyAATlAd2AZ0C/CXA3aq4s+rrD/DBGRZJTIoaETyk6b2WDgO5ElauA+2VlAcYlzYrfWcUcREakTNR6EPux+Oim7jK4L76/7FIAzjusQbxARkTqSyDmC/ykzeQQwGNgcWaIGrKTEGf30XABO7K7BZ0QkNSRyjqB5medFBOcM/h5NnIZt9dY9ADTPzmRwrg4NiUhqqLIQhDeSNXP3H9ZTngbt0j/MBOAXF/ePOYmISN2p9ByBmWW6ezHBoaC0t3jDDrbs3gfAyOOOjDmNiEjdqWqPYBZBEZhvZi8BfwP2HFjo7s9HnK1BGfNMcG7g95cPIicrI+Y0IiJ1J5FzBG2ArQTjCh+4n8CBtCkEhcUlrNv2GQBn9VOX0yKSWqoqBEeGVwwt4vMCcECDGje4vtx0Zh8aZdb4ilsRkQatqkKQATQjsUHoRUQkSVVVCDa6+531lqQBW5C/A4CiEtU/EUk9VR3nUHc6oUv/8A4AJ3bTTWQiknqqKgQj6y1FEmiencnwHm3jjiEiUucqLQTuvq0+gzRUW3bvo6jEufykXI1GJiIpSZfAVGPJhp0AtG2a1sM0i0gKUyGoxnfDG8mG99RhIRFJTSoEVSgqLmHP/mIAju/UMuY0IiLRUCGowpKNwWGh/xrcWecHRCRlqRBUYsWmXZz/4NsAXDiwc8xpRESio0JQib/PWw/AkK6tOamH7h8QkdSlQlCJVxd9DMCz3z6J7Ez1NioiqSuR3kfTUusmWRS1aawiICIpT3sElfh0byFd2zSNO4aISOQiLQRmNsrMlpnZCjMbW8Hyr5vZgvAxw8wGRJknUQ9NWcGqLXvYta8o7igiIpGLrBCE4x0/BJwN9AUuN7O+5ZqtBr7o7v2Bu4DxUeVJVFFxCfdNXgbAVV/oGnMaEZHoRblHMBRY4e6r3H0/MBG4oGwDd5/h7tvDyZlAlwjzJGT5J7sBuO70Xlw0KPY4IiKRi7IQdAbWlZnOD+dV5lvAKxUtMLPRZjbHzOZs3ry5DiMerKTEOeeBNwHo3aF5ZO8jItKQRFkIEh7ZzMy+RFAIbqloubuPd/c8d89r3759HUY86D0Y+ZtpAHRp3Zjz+neM5H1ERBqaKC8fzQeOLjPdBdhQvpGZ9QceA852960R5qnS799YweotewB44XsnY6YuJUQkPUS5RzAb6G1m3c2sEXAZ8FLZBmaWCzwPfMPdl0eYpVqTFm4EYNFPz6J98+w4o4iI1KvI9gjcvcjMrgUmAxnAE+6+2MzGhMvHAbcBbYGHw2/gRe6eF1Wmynz4yS4++HgXAM2ydY+diKSXSD/13H0SMKncvHFlnl8DXBNlhkRc89QcAK4+uVu8QUREYpD2dxa7O2u27qVTyxxuP69f3HFEROpd2heCbXv2A9CldZOYk4iIxCPtC8GB61nPHaDLRUUkPaV9IZi2LLhBzSu8w0FEJPWlfSGY/dE2AL7YJ5ob1UREGrq0LwTrP/0MgA4tcmJOIiISj7QvBJlHGP27tKRxIw1AIyLpKe0LgYhIulMhEBFJcyoEIiJpLu0LQYnr0lERSW9pXQhKSpxpyzdTWFwSdxQRkdikdSEY/XTQ2dxRLXXpqIikr7QuBK8v3QTAvV/tH3MSEZH4pG0h+GRnAQBXfaEbRzbXHoGIpK+0LQS/fOUDAHq0bxpzEhGReKVtIZi2POhs7sqTusacREQkXmlZCNZs3cPWPfvJyTqCI47QIPUikt7SshBc+NDbAPzf2cfFnEREJH5pVwj27Cti+95CAL45XIeFRETSrhD8ZfY6ICgCZjosJCKSGXeA+nbgstHrR/aOOYlI6issLCQ/P5+CgoK4o6SNnJwcunTpQlZWVsKvSatCMH/dp/xh+ioAmuek1aaLxCI/P5/mzZvTrVs37YHXA3dn69at5Ofn071794Rfl1aHhpZu3AnA1Sd3IztTA9GIRK2goIC2bduqCNQTM6Nt27Y13gNLq0JwwHdO7Rl3BJG0oSJQvw7n952WhUBERD6XVoXgjQ82xR1BRGLwwgsvYGZ88MEHpfOmTp3Kueeee1C7q666iueeew4ITnSPHTuW3r17c/zxxzN06FBeeeWVWme555576NWrF8cccwyTJ0+usM38+fMZNmwYAwcOJC8vj1mzZgHw0Ucf0bhxYwYOHMjAgQMZM2ZMrfNAmp0sLgrHHWjXrFHMSUSkPk2YMIFTTjmFiRMncscddyT0mp/85Cds3LiRRYsWkZ2dzSeffMK0adNqlWPJkiVMnDiRxYsXs2HDBs444wyWL19ORsbB5yxvvvlmbr/9ds4++2wmTZrEzTffzNSpUwHo2bMn8+fPr1WO8tKqEOzdX8wJnVuSmZFWO0IiDcJP/7mYJRt21uk6+3Zqwe3n9auyze7du3n77beZMmUK559/fkKFYO/evTz66KOsXr2a7OxsADp06MAll1xSq7wvvvgil112GdnZ2XTv3p1evXoxa9Yshg8fflA7M2PnzuB3tWPHDjp16lSr961O2hSCkhLn3dXb6H1ks7ijiEg9+sc//sGoUaPo06cPbdq0Yd68eQwePLjK16xYsYLc3FxatGhR7fp/8IMfMGXKlEPmX3bZZYwdO/ageevXr2fYsGGl0126dGH9+vWHvPZ3v/sdZ511FjfddBMlJSXMmDGjdNnq1asZNGgQLVq04O6772bEiBHVZqxO2hSC4nBg4uM6Vv+HFZG6V90396hMmDCBG2+8EQg+nCdMmMDgwYMrvbqmplfd/Pa3v024rVcwQHpF7/fII4/w29/+losvvpi//vWvfOtb3+L111+nY8eOrF27lrZt2zJ37lwuvPBCFi9enFDBqkqkhcDMRgH3AxnAY+7+i3LLLVx+DrAXuMrd50WZqU8H7RGIpIutW7fyxhtvsGjRIsyM4uJizIx7772Xtm3bsn379oPab9u2jXbt2tGrVy/Wrl3Lrl27aN68eZXvUZM9gi5durBu3brS6fz8/AoP+/zpT3/i/vvvB+BrX/sa11xzDQDZ2dmlh6qGDBlCz549Wb58OXl5eQn8Nqrg7pE8CD78VwI9gEbA+0Dfcm3OAV4BDBgGvFvdeocMGeKHY+nGHd71lpf9gdeXH9brRaTmlixZEuv7jxs3zkePHn3QvFNPPdWnT5/uBQUF3q1bt9KMH330kefm5vqnn37q7u4//OEP/aqrrvJ9+/a5u/uGDRv86aefrlWeRYsWef/+/b2goMBXrVrl3bt396KiokPaHXvssT5lyhR3d3/99dd98ODB7u6+adOm0vYrV670Tp06+datWw95fUW/d2COV/K5GuUewVBghbuvAjCzicAFwJIybS4AngpDzjSzVmbW0d031nWYlZv2ANCjvfYIRNLFhAkTDvlWfvHFF/Pss88yYsQInnnmGa6++moKCgrIysriscceo2XLlgDcfffd3HrrrfTt25ecnByaNm3KnXfeWas8/fr145JLLqFv375kZmby0EMPlV4xdM011zBmzBjy8vJ49NFHueGGGygqKiInJ4fx48cDMH36dG677TYyMzPJyMhg3LhxtGnTplaZAMwrOGZVF8zsq8Aod78mnP4GcJK7X1umzcvAL9z9rXD6P8At7j6n3LpGA6MBcnNzh6xZs6bGeeau2c7jb63i1q/0pVOrxoe7WSJSA0uXLuW44zTuR32r6PduZnPdvcJjSFHuEVR0xqV81UmkDe4+HhgPkJeXd1iVa0jX1gzpOuRwXioiktKivKA+Hzi6zHQXYMNhtBERkQhFWQhmA73NrLuZNQIuA14q1+Yl4JsWGAbsiOL8gIjEJ6rDz1Kxw/l9R3ZoyN2LzOxaYDLBFURPuPtiMxsTLh8HTCK4cmgFweWjV0eVR0TqX05ODlu3blVX1PXEw/EIcnJyavS6yE4WRyUvL8/nzJlTfUMRiZ1GKKt/lY1QFtfJYhFJc1lZWTUaKUviod7XRETSnAqBiEiaUyEQEUlzSXey2Mw2AzW/tTjQDthSh3GSgbY5PWib00Nttrmru7evaEHSFYLaMLM5lZ01T1Xa5vSgbU4PUW2zDg2JiKQ5FQIRkTSXboVgfNwBYqBtTg/a5vQQyTan1TkCERE5VLrtEYiISDkqBCIiaS4lC4GZjTKzZWa2wszGVrDczOyBcPkCMxscR866lMA2fz3c1gVmNsPMBsSRsy5Vt81l2p1oZsXhqHlJLZFtNrPTzGy+mS02s2n1nbGuJfBvu6WZ/dPM3g+3Oal7MTazJ8xsk5ktqmR53X9+VTaYcbI+CLq8Xgn0ABoB7wN9y7U5B3iFYIS0YcC7ceeuh23+AtA6fH52OmxzmXZvEHR5/tW4c9fD37kVwbjgueH0kXHnrodt/hHwy/B5e2Ab0Cju7LXY5lOBwcCiSpbX+edXKu4RDAVWuPsqd98PTAQuKNfmAuApD8wEWplZx/oOWoeq3WZ3n+Hu28PJmQSjwSWzRP7OANcBfwc21We4iCSyzVcAz7v7WgB3T/btTmSbHWhuwYAHzQgKQVH9xqw77j6dYBsqU+efX6lYCDoD68pM54fzatommdR0e75F8I0imVW7zWbWGbgIGFePuaKUyN+5D9DazKaa2Vwz+2a9pYtGItv8IHAcwTC3C4Eb3L2kfuLFos4/v1JxPIKKhkEqf41sIm2SScLbY2ZfIigEp0SaKHqJbPPvgFvcvThFRsdKZJszgSHASKAx8I6ZzXT35VGHi0gi23wWMB84HegJ/NvM3nT3nRFni0udf36lYiHIB44uM92F4JtCTdskk4S2x8z6A48BZ7v71nrKFpVEtjkPmBgWgXbAOWZW5O7/qJeEdS/Rf9tb3H0PsMfMpgMDgGQtBIls89XALzw4gL7CzFYDxwKz6idivavzz69UPDQ0G+htZt3NrBFwGfBSuTYvAd8Mz74PA3a4+8b6DlqHqt1mM8sFnge+kcTfDsuqdpvdvbu7d3P3bsBzwPeSuAhAYv+2XwRGmFmmmTUBTgKW1nPOupTINq8l2APCzDoAxwCr6jVl/arzz6+U2yNw9yIzuxaYTHDFwRPuvtjMxoTLxxFcQXIOsALYS/CNImkluM23AW2Bh8NvyEWexD03JrjNKSWRbXb3pWb2KrAAKAEec/cKL0NMBgn+ne8CnjSzhQSHTW5x96TtntrMJgCnAe3MLB+4HciC6D6/1MWEiEiaS8VDQyIiUgMqBCIiaU6FQEQkzakQiIikORUCEZE0p0IgDVLYW+j8Mo9uVbTdXQfv96SZrQ7fa56ZDT+MdTxmZn3D5z8qt2xGbTOG6znwe1kU9rjZqpr2A83snLp4b0ldunxUGiQz2+3uzeq6bRXreBJ42d2fM7MzgV+5e/9arK/Wmapbr5n9CVju7j+rov1VQJ67X1vXWSR1aI9AkoKZNTOz/4Tf1hea2SE9jZpZRzObXuYb84hw/plm9k742r+ZWXUf0NOBXuFr/ydc1yIzuzGc19TM/hX2f7/IzC4N5081szwz+wXQOMzx53DZ7vDnX8p+Qw/3RC42swwzu8/MZlvQx/x3Evi1vEPY2ZiZDbVgnIn3wp/HhHfi3glcGma5NMz+RPg+71X0e5Q0FHff23roUdEDKCboSGw+8ALBXfAtwmXtCO6qPLBHuzv8+b/Aj8PnGUDzsO10oGk4/xbgtgre70nC8QqArwHvEnTethBoStC98WJgEHAx8GiZ17YMf04l+PZdmqlMmwMZLwL+FD5vRNCLZGNgNHBrOD8bmAN0ryDn7jLb9zdgVDjdAsgMn58B/D18fhXwYJnX/xy4MnzeiqAPoqZx/731iPeRcl1MSMr4zN0HHpgwsyzg52Z2KkHXCZ2BDsDHZV4zG3gibPsPd59vZl8E+gJvh11rNCL4Jl2R+8zsVmAzQQ+tI4EXPOjADTN7HhgBvAr8ysx+SXA46c0abNcrwANmlg2MAqa7+2fh4aj+9vkoai2B3sDqcq9vbGbzgW7AXODfZdr/ycx6E/REmVXJ+58JnG9mN4XTOUAuyd0fkdSSCoEki68TjD41xN0Lzewjgg+xUu4+PSwUXwGeNrP7gO3Av9398gTe44fu/tyBCTM7o6JG7r7czIYQ9Pdyj5m95u53JrIR7l5gZlMJuk6+FJhw4O2A69x9cjWr+MzdB5pZS+Bl4PvAAwT97Uxx94vCE+tTK3m9ARe7+7JE8kp60DkCSRYtgU1hEfgS0LV8AzPrGrZ5FHicYLi/mcDJZnbgmH8TM+uT4HtOBy4MX9OU4LDOm2bWCdjr7s8Avwrfp7zCcM+kIhMJOgobQdCZGuHP7x54jZn1Cd+zQu6+A7geuCl8TUtgfbj4qjJNdxEcIjtgMnCdhbtHZjaosveQ9KFCIMniz0Cemc0h2Dv4oII2pwHzzew9guP497v7ZoIPxglmtoCgMBybyBu6+zyCcwezCM4ZPObu7wEnALPCQzQ/Bu6u4OXjgQUHThaX8xrBuLSvezD8IgTjRCwB5lkwaPkfqGaPPczyPkHXzPcS7J28TXD+4IApQN8DJ4sJ9hyywmyLwmlJc7p8VEQkzWmPQEQkzakQiIikORUCEZE0p0IgIpLmVAhERNKcCoGISJpTIRARSXP/HwpkPYFG7HfqAAAAAElFTkSuQmCC"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p><a name="26"></a></p>
<h3 id="Naive-Bayes">Naive Bayes<a class="anchor-link" href="#Naive-Bayes">&#182;</a></h3>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">naive_bayes_cls</span> <span class="o">=</span> <span class="n">clf</span> <span class="o">=</span> <span class="n">GaussianNB</span><span class="p">()</span>
<span class="n">under_naive_bayes_model</span> <span class="o">=</span> <span class="n">naive_bayes_cls</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
<span class="n">prediction</span> <span class="o">=</span> <span class="n">under_naive_bayes_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x_test</span><span class="p">)</span>
<span class="n">print_stats</span><span class="p">(</span><span class="s2">&quot;Under-sampled Naive Bayes&quot;</span><span class="p">,</span> <span class="n">y_test</span><span class="p">,</span> <span class="n">prediction</span><span class="p">)</span>
<span class="n">show_roc_curve</span><span class="p">(</span><span class="n">under_naive_bayes_model</span><span class="p">,</span> <span class="n">x_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Under-sampled Naive Bayes metrics:
Under-sampled
Naive Bayes
Accuracy: 0.76
Precision: 0.7660389610389611
Recall: 0.6785714285714286
F1 score: 0.76
Receiver operating characteristic (ROC) curve:
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h4 id="Over-sampling">Over-sampling<a class="anchor-link" href="#Over-sampling">&#182;</a></h4>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">over_naive_bayes_model</span> <span class="o">=</span> <span class="n">naive_bayes_cls</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">over_x_train</span><span class="p">,</span> <span class="n">over_y_train</span><span class="p">)</span>
<span class="n">prediction</span> <span class="o">=</span> <span class="n">over_naive_bayes_model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">over_x_test</span><span class="p">)</span>
<span class="n">print_stats</span><span class="p">(</span><span class="s2">&quot;Over-sampled Naive Bayes&quot;</span><span class="p">,</span> <span class="n">over_y_test</span><span class="p">,</span> <span class="n">prediction</span><span class="p">)</span>
<span class="n">show_roc_curve</span><span class="p">(</span><span class="n">over_naive_bayes_model</span><span class="p">,</span> <span class="n">over_x_test</span><span class="p">,</span> <span class="n">over_y_test</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Over-sampled Naive Bayes metrics:
Over-sampled
Naive Bayes
Accuracy: 0.7958868894601543
Precision: 0.7247966706370705
Recall: 0.8619979402677652
F1 score: 0.8083051665861903
Receiver operating characteristic (ROC) curve:
</pre>
</div>
</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p><a name="3"></a></p>
<h2 id="Results">Results<a class="anchor-link" href="#Results">&#182;</a></h2><p>Following the tests done and with the statistcs already filled, the multiple metrics are compared in bar plots by the same order used before: accuracy, precision, recall and f1 score!
For each ML Algorithm Model we will have the over-sampled statistic, in a darker color, followed by the under-sampled statistic, in a lighter color, because, most of the time, the values will be higher for the over</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">plot_stat</span><span class="p">(</span><span class="n">stat_name</span><span class="p">):</span>
    <span class="n">colot_pallet1</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;#e5571f&quot;</span><span class="p">,</span> <span class="s2">&quot;#f0960f&quot;</span><span class="p">,</span> <span class="s2">&quot;#8e00e6&quot;</span><span class="p">,</span> <span class="s2">&quot;#2c83ba&quot;</span><span class="p">,</span> <span class="s2">&quot;#43a863&quot;</span><span class="p">,</span> <span class="s2">&quot;#e5ba10&quot;</span><span class="p">]</span>
    <span class="n">colot_pallet2</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;#e57d25&quot;</span><span class="p">,</span> <span class="s2">&quot;#f3ab3f&quot;</span><span class="p">,</span> <span class="s2">&quot;#935ba4&quot;</span><span class="p">,</span> <span class="s2">&quot;#3a96d1&quot;</span><span class="p">,</span> <span class="s2">&quot;#4db86e&quot;</span><span class="p">,</span> <span class="s2">&quot;#f2cf45&quot;</span><span class="p">]</span>
    <span class="n">stat_name_sorted_stats</span> <span class="o">=</span> <span class="nb">dict</span><span class="p">(</span><span class="nb">sorted</span><span class="p">(</span><span class="n">stats</span><span class="o">.</span><span class="n">items</span><span class="p">(),</span><span class="n">key</span><span class="o">=</span><span class="k">lambda</span> <span class="n">item</span><span class="p">:</span><span class="n">item</span><span class="p">[</span><span class="mi">1</span><span class="p">][</span><span class="s2">&quot;Over-sampled&quot;</span><span class="p">][</span><span class="n">stat_name</span><span class="p">],</span> <span class="n">reverse</span><span class="o">=</span><span class="kc">True</span><span class="p">))</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span> <span class="o">=</span> <span class="p">(</span><span class="mi">12</span><span class="p">,</span> <span class="mi">8</span><span class="p">))</span>
    <span class="n">X_axis</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">stat_name_sorted_stats</span><span class="o">.</span><span class="n">keys</span><span class="p">()))</span>
    <span class="n">over_stat_name_sorted_values</span> <span class="o">=</span> <span class="p">[</span><span class="n">item</span><span class="p">[</span><span class="s2">&quot;Over-sampled&quot;</span><span class="p">][</span><span class="n">stat_name</span><span class="p">]</span><span class="o">*</span><span class="mi">100</span> <span class="k">for</span> <span class="n">item</span> <span class="ow">in</span> <span class="n">stat_name_sorted_stats</span><span class="o">.</span><span class="n">values</span><span class="p">()]</span>
    <span class="n">under_stat_name_sorted_values</span> <span class="o">=</span> <span class="p">[</span><span class="n">item</span><span class="p">[</span><span class="s2">&quot;Under-sampled&quot;</span><span class="p">][</span><span class="n">stat_name</span><span class="p">]</span><span class="o">*</span><span class="mi">100</span> <span class="k">for</span> <span class="n">item</span> <span class="ow">in</span> <span class="n">stat_name_sorted_stats</span><span class="o">.</span><span class="n">values</span><span class="p">()]</span>
    <span class="n">bar_plot1</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">X_axis</span> <span class="o">-</span> <span class="mf">0.2</span><span class="p">,</span> <span class="n">over_stat_name_sorted_values</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">colot_pallet1</span><span class="p">,</span> <span class="n">width</span><span class="o">=</span><span class="mf">0.5</span><span class="p">)</span> 
    <span class="n">bar_plot2</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">X_axis</span> <span class="o">+</span> <span class="mf">0.2</span><span class="p">,</span> <span class="n">under_stat_name_sorted_values</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">colot_pallet2</span><span class="p">,</span> <span class="n">width</span><span class="o">=</span><span class="mf">0.5</span><span class="p">)</span> 
    <span class="k">for</span> <span class="n">idx</span><span class="p">,</span><span class="n">rect</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">bar_plot1</span><span class="p">):</span>
        <span class="n">height</span> <span class="o">=</span> <span class="n">rect</span><span class="o">.</span><span class="n">get_height</span><span class="p">()</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">text</span><span class="p">(</span><span class="n">rect</span><span class="o">.</span><span class="n">get_x</span><span class="p">()</span> <span class="o">+</span> <span class="n">rect</span><span class="o">.</span><span class="n">get_width</span><span class="p">()</span><span class="o">/</span><span class="mf">2.</span><span class="p">,</span> <span class="mf">0.95</span><span class="o">*</span><span class="n">height</span><span class="p">,</span>
            <span class="s2">&quot;</span><span class="si">{:.1f}</span><span class="s2">%&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">over_stat_name_sorted_values</span><span class="p">[</span><span class="n">idx</span><span class="p">]),</span>
            <span class="n">ha</span><span class="o">=</span><span class="s1">&#39;center&#39;</span><span class="p">,</span> <span class="n">va</span><span class="o">=</span><span class="s1">&#39;bottom&#39;</span><span class="p">,</span> <span class="n">rotation</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>

    <span class="k">for</span> <span class="n">idx</span><span class="p">,</span><span class="n">rect</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">bar_plot2</span><span class="p">):</span>
        <span class="n">height</span> <span class="o">=</span> <span class="n">rect</span><span class="o">.</span><span class="n">get_height</span><span class="p">()</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">text</span><span class="p">(</span><span class="n">rect</span><span class="o">.</span><span class="n">get_x</span><span class="p">()</span> <span class="o">+</span> <span class="n">rect</span><span class="o">.</span><span class="n">get_width</span><span class="p">()</span><span class="o">/</span><span class="mf">2.</span><span class="p">,</span> <span class="mf">0.95</span><span class="o">*</span><span class="n">height</span><span class="p">,</span>
            <span class="s2">&quot;</span><span class="si">{:.1f}</span><span class="s2">%&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">under_stat_name_sorted_values</span><span class="p">[</span><span class="n">idx</span><span class="p">]),</span>
            <span class="n">ha</span><span class="o">=</span><span class="s1">&#39;center&#39;</span><span class="p">,</span> <span class="n">va</span><span class="o">=</span><span class="s1">&#39;bottom&#39;</span><span class="p">,</span> <span class="n">rotation</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
        
    <span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">(</span><span class="n">X_axis</span><span class="p">,</span> <span class="n">stat_name_sorted_stats</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">&quot;ML Algorithm Models&quot;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="n">stat_name</span><span class="o">.</span><span class="n">title</span><span class="p">()</span> <span class="o">+</span> <span class="s2">&quot; in %&quot;</span><span class="p">)</span> 
    <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Models ranked by &quot;</span> <span class="o">+</span> <span class="n">stat_name</span> <span class="o">+</span> <span class="s2">&quot;!&quot;</span><span class="p">)</span> 
    <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span> 
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>This function serves to plot any stat!</p>
<h3 id="Accuracy">Accuracy<a class="anchor-link" href="#Accuracy">&#182;</a></h3>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plot_stat</span><span class="p">(</span><span class="s2">&quot;accuracy&quot;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="Precision">Precision<a class="anchor-link" href="#Precision">&#182;</a></h3>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plot_stat</span><span class="p">(</span><span class="s2">&quot;precision&quot;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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xVszs7H7fbr247Bts/tmL5jJlY9cwvG7ZUfMuPT+83nl3eeZt3geP7jha+y32afZfr1dgSxsr913JD27ZOF7nf7rc86/T2No77VZq3f7OePc7KfvpeemuwEw5/mHqOzRj86D121ynlRXy6sXnEjVjCn02eZAugzbCIAeG+/Mq386nq4jt6CsU1cWvv0iA3Y7suTbIEkfhqFZkpqhQ0VHfnHIhctM+/yOxzfatmeX3ksCM8AXdjpppcvdfNhYNh+29FBan9zycD655eEfstqWVVdTzdwXHmLQXl+krmoR0++7khFH/7xwvigrZ70TL6R24Tze/MePWfTu63QauA79d/4s/Xf+LABv/+vXDNj9aGaMv4V5r0yg06B16PHpb5V6kyRplXn0DElSk+a9/BidBq9HRbfeVM2YStXMd3jlj1/lxfOOonrOdF694ASq585Y6fzlnbvRdcRmzHt5/DLTF055BYCO/YYya+I41j7sDBa/O4mF094o6fZI0gdhT7MkqUmzn7qbXptlQzM6DVqHjU67dsl9L553FOt+9fwVjp5RM38WUVZBeedu1FUvZt5rT9Av712uN+3Oyxly0MnZeOd8LDdRRl3VotJukCR9AIZmSdJK1VUtYt6rjzPkoJML21bPeZ+3//VrRnz+bGrmzuCt639JSnWQ6ui5yS70GLXdkrZznnuAzkM3oLJHXwA6D9uYl/9wHJ0GrUPXtUatbBWS1GYMzVI78dLbCzjmN88tuT1p2iK+f+gIpry/mFsmvE+HijLWGdiJC07ckF5dl33pLqqqY98fTmRxTR01tYmDt+vP9w8dAcAP/v4a456YwWYjunHx1zYE4B/3vsvMedWc8IkPdrQHrTnKOnRio9OvX+n9o779tyXXK3v0ZcTns+M6dxo0kvVOvGCl8/XYeMclh68DGLzvcbBv849xvaabP2UWz/zu7iW3F06by8hDtmTxjAW89/iblFWU0XlgDzb66s5Udl3xBFMPnHQ15Z0ribIgysvY5pzsTHyvXPEo7z/5Ft2G92X0ibsAMPW+l6met5i199ukdTZOaqcMzVI7scHQLjz4q+wHYbW1iQ2+8hAHbNOPl6cs4MdHjKSiPPjB31/jvH++yU+PHLnMvB0rg3//aHO6dS6nuqaOvX8wkb226MOooV145MXZPHzeWL70u+d59o15jBzUmSvueYd/fn/T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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="Recall">Recall<a class="anchor-link" href="#Recall">&#182;</a></h3>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plot_stat</span><span class="p">(</span><span class="s2">&quot;recall&quot;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="F1-score">F1 score<a class="anchor-link" href="#F1-score">&#182;</a></h3>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plot_stat</span><span class="p">(</span><span class="s2">&quot;f1_score&quot;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p><a name="4"></a></p>
<h2 id="Conclusion">Conclusion<a class="anchor-link" href="#Conclusion">&#182;</a></h2><p>The results differ very little from the ones suggested in the paper when using the under-sampling method to balance the data.
Using over-sampling yields, in general, better results tho ti might come at a cost since the computational effort for the model building is increased!
A ML algorithm trained on image datasets would probably lead to a better model for stroke prediction, though with some of these algorithms, on this dataset, we do get fairly good results!</p>

</div>
</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p><a name="5"></a></p>
<h2 id="References">References<a class="anchor-link" href="#References">&#182;</a></h2><p>[1] Accessed 20th of February, 2022 - <a href="https://thesai.org/Downloads/Volume12No6/Paper_62-Analyzing_the_Performance_of_Stroke_Prediction.pdf">https://thesai.org/Downloads/Volume12No6/Paper_62-Analyzing_the_Performance_of_Stroke_Prediction.pdf</a></p>

</div>
</div>
</body>







</html>
