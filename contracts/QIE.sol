// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";

/// @title QuantumIDExclusive (QIE)
/// @notice Soulbound identity token for Q-BOND Network
contract QuantumIDExclusive is ERC721URIStorage, Ownable {
    using Counters for Counters.Counter;

    Counters.Counter private _tokenIds;

    mapping(address => bool) public hasMinted;
    mapping(uint256 => uint256) public xp;
    mapping(uint256 => uint256) public trustScore;
    mapping(uint256 => uint256) public entropyDecay;
    mapping(uint256 => bool) public blacklisted;
    mapping(address => uint256) public qieOfUser; // Fast lookup mapping

    // Role separation for validators
    mapping(address => bool) public xpValidators;
    mapping(address => bool) public resurrectionValidators;

    // OPTIONAL ADD-ONS
    mapping(uint256 => uint256[]) public xpHistory; // XP History log
    mapping(uint256 => uint256) public lastXPUpdate; // For decay timer
    mapping(uint256 => uint256) public fieldScoreLambda; // Ξ Field Snapshot

    event QIEMinted(address indexed user, uint256 tokenId);
    event XPUpdated(uint256 tokenId, uint256 newXP);
    event XPLog(uint256 tokenId, uint256 oldXP, uint256 newXP, uint256 time); // 🧾 XP Log Event
    event TrustAdjusted(uint256 tokenId, uint256 newTrust);
    event SignalResurrected(uint256 tokenId);
    event QIEMetadataState(uint256 tokenId, uint256 xp, uint256 trust, uint256 lambda);

    modifier onlyOneMint() {
        require(!hasMinted[msg.sender], "QIE already minted.");
        _;
    }

    modifier onlyXPValidator() {
        require(xpValidators[msg.sender] || msg.sender == owner(), "Not XP validator.");
        _;
    }

    modifier onlyResurrectionValidator() {
        require(resurrectionValidators[msg.sender] || msg.sender == owner(), "Not Resurrection validator.");
        _;
    }

    constructor() ERC721("QuantumIDExclusive", "QIE") {}

    function mintQIE(string memory tokenURI) external onlyOneMint returns (uint256) {
        _tokenIds.increment();
        uint256 newId = _tokenIds.current();

        _safeMint(msg.sender, newId);
        _setTokenURI(newId, tokenURI);

        hasMinted[msg.sender] = true;
        qieOfUser[msg.sender] = newId;
        xp[newId] = 0;
        trustScore[newId] = 800;
        entropyDecay[newId] = 0;
        lastXPUpdate[newId] = block.timestamp;

        emit QIEMinted(msg.sender, newId);
        return newId;
    }

    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 firstTokenId,
        uint256 batchSize
    ) internal virtual override {
        super._beforeTokenTransfer(from, to, firstTokenId, batchSize);
        require(from == address(0) || to == address(0), "Soulbound: transfers disabled");
    }

    function updateXP(uint256 tokenId, uint256 newXP) external onlyXPValidator {
        require(!_isBlacklisted(tokenId) || msg.sender == owner(), "Blacklisted"); // Admin override
        uint256 oldXP = xp[tokenId];
        xp[tokenId] = newXP;
        xpHistory[tokenId].push(newXP);
        lastXPUpdate[tokenId] = block.timestamp;

        emit XPUpdated(tokenId, newXP);
        emit XPLog(tokenId, oldXP, newXP, block.timestamp);
        emit QIEMetadataState(tokenId, newXP, trustScore[tokenId], fieldScoreLambda[tokenId]);
    }

    function setTrustScore(uint256 tokenId, uint256 newScore) external onlyXPValidator {
        require(!_isBlacklisted(tokenId) || msg.sender == owner(), "Blacklisted"); // Admin override
        require(newScore <= 1000, "Max trust = 1000");
        trustScore[tokenId] = newScore;

        emit TrustAdjusted(tokenId, newScore);
        emit QIEMetadataState(tokenId, xp[tokenId], newScore, fieldScoreLambda[tokenId]);
    }

    function decayEntropy(uint256 tokenId, uint256 amount) external onlyXPValidator {
        entropyDecay[tokenId] += amount;
    }

    function autoDecayIfInactive(uint256 tokenId) external {
        require(block.timestamp > lastXPUpdate[tokenId] + 90 days, "Decay not triggered yet");
        entropyDecay[tokenId] += 10;
    }

    function decayTrustIfInactive(uint256 tokenId) external {
        require(block.timestamp > lastXPUpdate[tokenId] + 180 days, "Decay not yet triggered");
        trustScore[tokenId] -= 10;
    }

    function resurrectSignal(uint256 tokenId) external onlyResurrectionValidator {
        require(xp[tokenId] >= 100, "Not enough XP to resurrect");
        // Resurrection requires symbolic memory
        entropyDecay[tokenId] = 0;
        trustScore[tokenId] = 800;

        emit SignalResurrected(tokenId);
        emit QIEMetadataState(tokenId, xp[tokenId], 800, fieldScoreLambda[tokenId]);
    }

    function blacklist(uint256 tokenId) external onlyOwner {
        blacklisted[tokenId] = true;
    }

    function unblacklist(uint256 tokenId) external onlyOwner {
        blacklisted[tokenId] = false;
    }

    function whitelistXPValidator(address validator) external onlyOwner {
        xpValidators[validator] = true;
    }

    function whitelistResurrectionValidator(address validator) external onlyOwner {
        resurrectionValidators[validator] = true;
    }

    function removeValidator(address validator) external onlyOwner {
        xpValidators[validator] = false;
        resurrectionValidators[validator] = false;
    }

    function _isBlacklisted(uint256 tokenId) internal view returns (bool) {
        return blacklisted[tokenId];
    }

    function getQIE(address user) external view returns (uint256 tokenId, uint256 xpPoints, uint256 trust, uint256 decay, uint256 lambda) {
        require(hasMinted[user], "QIE not found");
        uint256 id = qieOfUser[user];
        return (id, xp[id], trustScore[id], entropyDecay[id], fieldScoreLambda[id]);
    }

    function getXPHistory(uint256 tokenId) external view returns (uint256[] memory) {
        return xpHistory[tokenId];
    }

    function hasValidatorRole(address user) external view returns (bool xpRole, bool resurrectionRole) {
        return (xpValidators[user], resurrectionValidators[user]);
    }

    function getTrustTier(uint256 tokenId) public view returns (string memory) {
        uint256 score = trustScore[tokenId];
        if (score >= 900) return unicode"Ξ Curator";
        if (score >= 800) return unicode"Λ Pathfinder";
        if (score >= 700) return unicode"Ψ Steward";
        return unicode"Σ Initiate";
    }

    function setLambdaScore(uint256 tokenId, uint256 lambdaValue) external onlyXPValidator {
        fieldScoreLambda[tokenId] = lambdaValue;
    }
}
