const express = require("express");
const router = express.Router();
const User = require("../model/user");
const authMiddleware = require("../middleware/authmiddleware"); // Assuming authentication middleware

// POST: Add news item to bookmarks
router.post("/bookmark", authMiddleware, async (req, res) => {
  const { title, description, url } = req.body;
  console.log("hhbhj");





  if (!title || !description || !url) {
    console.error('Missing fields:', { title, description, url }); // Log missing fields
    return res.status(400).json({ message: "Missing required fields" });
  }

  try {
    console.log('Request body:', req.body); // Log the request body for debugging
    console.log("rgtrgtr",req.user);
    const user = await User.findById(req.user.userId);
    console.log(user);
    if (!user) {
      console.error('User not found:', req.user.userId); // Log user lookup failure
      return res.status(404).json({ message: "User not found" });
    }

    const bookmarkExists = user.bookmarks.some((bookmark) => bookmark.url === url);
    if (bookmarkExists) {
      console.log('Bookmark already exists:', url); // Log existing bookmark check
      return res.status(400).json({ message: "Bookmark already exists" });
    }

    user.bookmarks.push({ title, description, url });
    await user.save();
    console.log('Bookmark added:', { title, description, url }); // Log successful addition
    console.log(user);
    res.status(200).json({ title, description, url });
  } catch (error) {
    console.error('Error in /bookmark route:', error); // Log the actual error
    res.status(500).json({ message: "Error adding bookmark" });
  }
});


// GET: Fetch the list of bookmarks for the user
router.get("/bookmarksshown", authMiddleware, async (req, res) => {
  try {
    const user = await User.findById(req.user.userId); // Fetch user by ID from the token
    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }
    res.status(200).json(user.bookmarks); // Return the list of bookmarks
    console.log("ram ram",user.bookmarks);
  } catch (error) {
    console.error("Error fetching bookmarks:", error);
    res.status(500).json({ message: "Error fetching bookmarks" });
  }
});


router.delete('/removeBookmark', authMiddleware, async (req, res) => {
  try {
    const { bookmarkId } = req.body; // Expecting the bookmark ID from the request body
    const user = await User.findById(req.user.userId); // Get the authenticated user

    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    // Filter out the bookmark to be removed
    user.bookmarks = user.bookmarks.filter((bookmark) => bookmark._id.toString() !== bookmarkId);

    await user.save(); // Save the updated user
    res.status(200).json({ message: 'Bookmark removed successfully', bookmarks: user.bookmarks });
  } catch (error) {
    console.error('Error removing bookmark:', error);
    res.status(500).json({ message: 'Error removing bookmark' });
  }
});


module.exports=router;