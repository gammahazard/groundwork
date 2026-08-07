"use strict";
/* The API reference's shared registry. Loaded FIRST; every group file pushes
 * itself in, and tabs/api/docs.js renders whatever is here.
 *
 * ONE FILE PER GROUP, because this is the part of the app most likely to be
 * edited by someone documenting a route they just added — and a single 900-line
 * catalogue is exactly the shape that makes people not bother. A new group is a
 * new file plus one <script> tag; nothing else knows.
 *
 * CURATED, NOT GENERATED. The server has 115 endpoints and dumping all of them
 * would be a list, not documentation: the reader could not tell which four
 * matter, and each would carry a description no better than its path.
 */
const API_GROUPS = [];

/* Register a group. `order` keeps the sidebar stable no matter which <script>
 * the browser finishes first — load order is not a promise. */
function apiGroup(g) { API_GROUPS.push(g); }

window.API_GROUPS = API_GROUPS;
window.apiGroup = apiGroup;
