# Phishing Dataset
Paper: https://www.sciencedirect.com/science/article/abs/pii/S0020025519300763?via%3Dihub

### Filtered Features:
No. Identifier Value type Description

**URL features:**
1. UrlLength (Discrete): Counts the total characters in the webpage URL.
2. NumNumericChars (Discrete): Counts the number of numeric characters in the webpage URL.
3. NumSensitiveWords (Discrete): Counts the number of sensitive words (i.e., “secure”, “account”, “webscr”, “login”,
“ebayisapi”, “signin”, “banking”, “confirm”) in webpage URL.

**Hyperlinks in the web page:**
4. PctExtHyperlinks (Continuous): Counts the proportion of external hyperlinks ('a' tags) in webpage HTML source code.
5. PctNullSelfRedirectHyperlinks (Continuous): Counts the percentage of hyperlinks ('a' tags) containing empty value, self-redirect value such as url starting with “#”, the URL of current webpage, or some abnormal value such as “file://E:/”.

**Source code of the web page:**
6. FrequentDomainNameMismatch (Binary): Checks if the most frequent domain name in HTML source code hyperlinks ('a' tags) does not match the origin webpage URL domain name.
7. SubmitInfoToEmail (Binary): Check if HTML source code contains the HTML “mailto” function.

**Range features**
The following features represent a range of certain rates.
8. PctExtResourceUrlsRT (Categorical / Range): Counts the proportion of external resource URLs in webpage HTML source code, out of all the URL in recsources. Resources are tags from ['img', 'video', 'audio']. 
9. ExtMetaScriptLinkRT (Categorical / Range): Counts proportion of meta, script and link tags containing external URL in the
attributes, out of all the links available in these tags.
10. PctExtNullSelfRedirectHyperlinksRT (Categorical / Range): Simply the addition of the two features - PctExtHyperlinks, PctNullSelfRedirectHyperlinks.


# Golden Constraints of the dataset
*  NumNumericChars < UrlLength


### The rest of the features:
No. Identifier Value type Description
1 NumDots Discrete Counts the number of dots in webpage URL [1,31].
2 SubdomainLevel Discrete Counts the level of subdomain in webpage URL [13,30]
3 PathLevel Discrete Counts the depth of the path in webpage URL [9].

5 NumDash Discrete Counts the number of “-” in webpage URL [9,16,17,30].
6 NumDashInHostname Discrete Counts the number of “-” in hostname part of webpage URL [9,16,17,30].
7 AtSymbol Binary Checks if “@” symbol exist in webpage URL [9,13,16,17,30].
8 TildeSymbol Binary Checks if “ ∼ ” symbol exist in webpage URL [8].
9 NumUnderscore Discrete Counts the number of “_” in webpage URL [9].
10 NumPercent Discrete Counts the number of “%” in webpage URL [8].
11 NumQueryComponents Discrete Counts the number of query parts in webpage URL [8].
12 NumAmpersand Discrete Counts the number of “&” in webpage URL [8].
13 NumHash Discrete Counts the number of “#” in webpage URL [8].
15 NoHttps Binary Checks if HTTPS exist in webpage URL [9,13,15–17].
16 RandomString Binary Checks if random strings exist in webpage URL [8].
17 IpAddress Binary Checks if IP address is used in hostname part of webpage URL [10,13,16,17,30].
18 DomainInSubdomains Binary Checks if TLD or ccTLD is used as part of subdomain in webpage URL [30].
19 DomainInPaths Binary Checks if TLD or ccTLD is used in the path of webpage URL [13,17,30].
20 HttpsInHostname Binary Checks if HTTPS in obfuscated in hostname part of webpage URL.
21 HostnameLength Discrete Counts the total characters in hostname part of webpage URL [15].
22 PathLength Discrete Counts the total characters in path of webpage URL [15].
23 QueryLength Discrete Counts the total characters in query part of webpage URL [15].
24 DoubleSlashInPath Binary Checks if “//” exist in the path of webpage URL [23].

26 EmbeddedBrandName Binary Checks if brand name appears in subdomains and path of webpage URL [30]. Brand
name here is assumed as the most frequent domain name in the webpage HTML
content.
28 PctExtResourceUrls Continuous Counts the percentage of external resource URLs in webpage HTML source
code [13,16,17].
29 ExtFavicon Binary Checks if the favicon is loaded from a domain name that is different from the webpage
URL domain name [17].
30 InsecureForms Binary Checks if the form action attribute contains a URL without HTTPS protocol [30].
31 RelativeFormAction Binary Checks if the form action attribute contains a relative URL [30].
32 ExtFormAction Binary Checks if the form action attribute contains a URL from an external domain [16,17].
33 AbnormalFormAction Categorical Check if the form action attribute contains a “#”, “about:blank”, an empty string, or
“javascript:true” [17].


36 FakeLinkInStatusBar Binary Checks if HTML source code contains JavaScript command onMouseOver to display a
fake URL in the status bar [16,17].
37 RightClickDisabled Binary Checks if HTML source code contains JavaScript command to disable right click
function [16,17].
38 PopUpWindow Binary Checks if HTML source code contains JavaScript command to launch pop-ups [8,16,17].
40 IframeOrFrame Binary Checks if iframe or frame is used in HTML source code [17].
41 MissingTitle Binary Checks if the title tag is empty in HTML source code [8].
42 ImagesOnlyInForm Binary Checks if the form scope in HTML source code contains no text at all but images only.
43 SubdomainLevelRT Categorical Counts the number of dots in hostname part of webpage URL. Apply rules and
thresholds to generate value [16].
44 UrlLengthRT Categorical Counts the total characters in the webpage URL. Apply rules and thresholds to generate
value [16,17].

46 AbnormalExtFormActionR Categorical Check if the form action attribute contains a foreign domain, “about:blank” or an
empty string. Apply rules to generate value [17].

